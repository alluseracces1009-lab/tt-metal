// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/semaphore_binding_token.h"  // SemScope + SemaphoreBindingToken

#ifdef COMPILE_FOR_TRISC
#include "api/compute/experimental/semaphore_compute_impl.h"
#else
#include "api/dataflow/semaphore_dm_impl.h"
#endif

/**
 * @brief Semaphore synchronization primitive for programmable cores.
 *
 * The host picks the access mechanism (SemScope) from where the semaphore's binder kernels run and
 * delivers it in a generated binding token; construct with CTAD, `Semaphore s(sem::name);`.
 *
 * DM builds expose local operations plus the NoC operations (remote up, set/relay/inc multicast).
 *
 * Blackhole UNPACK/PACK builds expose the local operations only, on the Tensix hardware (Sync Unit)
 * semaphore, and require SemScope::COMPUTE_ATOMIC; any other scope is a compile error, so a compute
 * kernel cannot reach a semaphore through a non-atomic path. Compute rules a kernel author must know:
 *  - The value is 0..15. More than 15 outstanding credits lose posts silently; a producer that gates
 *    each up() with wait_not_full() can never get there.
 *  - Its capacity (hardware Max) is SemaphoreAdvancedOptions::max_value, the ring depth in credits;
 *    wait_not_full() blocks while the value is at capacity. Default capacity is 15.
 *  - It starts every kernel at 0 (seeded by compute_kernel_hw_startup on PACK) and the kernel must
 *    leave it balanced at 0; the host rejects a nonzero initial_value and a second compute semaphore.
 *  - up()/down() are ordered after this thread's engine work (packer/unpacker), and wait()/wait_min()
 *    order this thread's *subsequent engine instructions* after the condition, not RISC loads. A
 *    kernel that reads the buffer with the RISC must poll value() instead.
 *  - MATH and isolate-SFPU builds expose no synchronization methods.
 * Details and instruction sequences: semaphore_compute_impl.h.
 */
template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX, SemScope SCOPE = SemScope::LOCAL_NONATOMIC>
class Semaphore {
    template <ProgrammableCoreType, SemScope>
    friend class Semaphore;

public:
    template <std::uint32_t SEM_ID, SemScope TOK_SCOPE>
    explicit __attribute__((always_inline)) Semaphore(SemaphoreBindingToken<SEM_ID, TOK_SCOPE>) :
        l1_offset_(semaphore_detail::sem_l1_offset<core_type, SCOPE>(SEM_ID)) {
        static_assert(
            TOK_SCOPE == SCOPE,
            "construct a bound semaphore with CTAD: `Semaphore s(sem::name);`; spelling "
            "`Semaphore<>` fixes the scope to LOCAL_NONATOMIC");
    }

    explicit __attribute__((always_inline)) Semaphore(std::uint32_t semaphore_id) :
        l1_offset_(semaphore_detail::sem_l1_offset<core_type, SCOPE>(semaphore_id)) {
#ifdef COMPILE_FOR_TRISC
        // A runtime id is invisible to the host semaphore census, so the host cannot know a
        // compute thread touches this word and cannot force every participant onto the same
        // mechanism -- a DM kernel binding the same word would resolve to LOCAL_NONATOMIC and its
        // plain read-modify-write would drop this thread's atomic update. Compute therefore takes
        // host-generated binding tokens only.
        static_assert(
            semaphore_detail::always_false<SCOPE>,
            "a compute semaphore cannot be built from a runtime semaphore id: bind it in the "
            "ProgramSpec and construct it from the generated token instead -- `Semaphore "
            "s(sem::name);`");
#else
        static_assert(
            SCOPE == SemScope::LOCAL_NONATOMIC,
            "a runtime semaphore id has no host-resolved mechanism and is LOCAL_NONATOMIC only");
#endif
    }

#if !defined(COMPILE_FOR_TRISC) || (defined(ARCH_BLACKHOLE) && (defined(TRISC_UNPACK) || defined(TRISC_PACK)))
    /**
     * @brief Increment the semaphore by `value`. Never blocks the caller.
     *
     * DM_LOCAL_CACHED: RISC atomic add. EXTERNAL: self-targeted NoC atomic. LOCAL_NONATOMIC: L1
     * read-modify-write (not atomic; the host picks it only for a single binder). COMPUTE_ATOMIC:
     * `value` SEMPOSTs, each ordered after this thread's packer/unpacker work, so a consumer that sees
     * the credit also sees the data (publish-after-data); value + outstanding credits must stay <= 15.
     *
     * @param value Amount to add.
     */
    __attribute__((always_inline)) void up(std::uint32_t value) {
        semaphore_detail::up<core_type, SCOPE>(l1_offset_, value);
    }

    /**
     * @brief Decrement the semaphore by `value`.
     *
     * DM: blocks until the value is at least `value`, then subtracts it (atomically for CACHED and
     * EXTERNAL). COMPUTE_ATOMIC: `value` SEMGETs ordered after this thread's engine work; does NOT wait
     * for sufficiency (SEMGET floors at 0), so pair it with wait_min() -- `wait_min(n); <read>; down(n)`.
     *
     * @param value Amount to subtract.
     */
    __attribute__((always_inline)) void down(std::uint32_t value) {
        semaphore_detail::down<core_type, SCOPE>(l1_offset_, value);
    }

    /**
     * @brief Block until the semaphore equals `value`. Does not modify it.
     *
     * DM: RISC poll. COMPUTE_ATOMIC: retires this thread's own posted up()/down() first, then RISC-polls.
     *
     * @param value Value to wait for.
     */
    __attribute__((always_inline)) void wait(std::uint32_t value) const {
        semaphore_detail::wait<SCOPE>(l1_offset_, value);
    }

    /**
     * @brief Block until the semaphore is at least `value`. Does not modify it.
     *
     * DM: RISC poll. COMPUTE_ATOMIC with value == 1: Tensix-side SEMWAIT -- the RISC returns at once
     * and this thread's next engine instructions (UNPACR/PACR) are held until the value is nonzero;
     * the fastest form. Other values: as wait().
     *
     * @param value Minimum value to wait for.
     */
    __attribute__((always_inline)) void wait_min(std::uint32_t value) const {
        semaphore_detail::wait_min<SCOPE>(l1_offset_, value);
    }

#ifdef COMPILE_FOR_TRISC
    /**
     * @brief Producer back-pressure (compute only). Block this thread's next engine instructions
     * (PACR/UNPACR) while the semaphore is at its capacity, SemaphoreAdvancedOptions::max_value. The RISC
     * returns at once. Canonical producer loop: `wait_not_full(); pack_tile(ring, slot); up(1);`.
     */
    __attribute__((always_inline)) void wait_not_full() const { semaphore_detail::wait_not_full<SCOPE>(l1_offset_); }
#endif

    /**
     * @brief Set the semaphore to `value`.
     *
     * DM: plain store. COMPUTE_ATOMIC: SEMINIT ordered after this thread's engine work; `value` <= 15.
     *
     * @param value New value.
     */
    __attribute__((always_inline)) void set(std::uint32_t value) { semaphore_detail::set<SCOPE>(l1_offset_, value); }

    /**
     * @brief The settled current value.
     *
     * DM: a fresh (cache-invalidated) read; a RISC-side write has already retired. COMPUTE_ATOMIC: first
     * retires this thread's own posted SEMPOST/SEMGET (tensix_sync), then reads the Sync Unit.
     *
     * @return Current semaphore value.
     */
    __attribute__((always_inline)) std::uint32_t value() const {
        return semaphore_detail::current<SCOPE>(l1_offset_);
    }
#endif

#ifndef COMPILE_FOR_TRISC
    /**
     * @brief Atomically increment the semaphore at (noc_x, noc_y) by `value` over the NoC. On a
     * DM_LOCAL_CACHED semaphore the only legal target is this node and the local atomic is used.
     */
    __attribute__((always_inline)) void up(
        const Noc& noc,
        std::uint32_t noc_x,
        std::uint32_t noc_y,
        std::uint32_t value,
        std::uint8_t vc = NOC_UNICAST_WRITE_VC) {
        semaphore_detail::up_remote<core_type, SCOPE>(l1_offset_, noc, noc_x, noc_y, value, vc);
    }

    /// @brief Write this semaphore's local value into `dst_sem` on core (noc_x, noc_y).
    template <ProgrammableCoreType dst_core_type = core_type, SemScope dst_scope = SemScope::LOCAL_NONATOMIC>
    void relay_unicast(
        const Noc& noc, const Semaphore<dst_core_type, dst_scope>& dst_sem, std::uint32_t noc_x, std::uint32_t noc_y) {
        semaphore_detail::relay_unicast<SCOPE, dst_scope>(l1_offset_, dst_sem.l1_offset_, noc, noc_x, noc_y);
    }

    /// @brief Multicast this semaphore's local value to the same semaphore on the rectangle of cores
    /// (NocOptions::MCAST_INCL_SRC includes the sender).
    template <NocOptions opts = NocOptions::DEFAULT>
    void set_multicast(
        const Noc& noc,
        std::uint32_t noc_x_start,
        std::uint32_t noc_y_start,
        std::uint32_t noc_x_end,
        std::uint32_t noc_y_end,
        std::uint32_t num_dests,
        bool linked = false) {
        semaphore_detail::set_multicast<opts, SCOPE>(
            l1_offset_, noc, noc_x_start, noc_y_start, noc_x_end, noc_y_end, num_dests, linked);
    }

    /// @brief Multicast this semaphore's local value into `dst_sem` on the rectangle of cores.
    template <
        NocOptions opts = NocOptions::DEFAULT,
        ProgrammableCoreType dst_core_type = core_type,
        SemScope dst_scope = SemScope::LOCAL_NONATOMIC>
    void relay_multicast(
        const Noc& noc,
        const Semaphore<dst_core_type, dst_scope>& dst_sem,
        std::uint32_t noc_x_start,
        std::uint32_t noc_y_start,
        std::uint32_t noc_x_end,
        std::uint32_t noc_y_end,
        std::uint32_t num_dests,
        bool linked = false) {
        semaphore_detail::relay_multicast<opts, SCOPE, dst_scope>(
            l1_offset_, dst_sem.l1_offset_, noc, noc_x_start, noc_y_start, noc_x_end, noc_y_end, num_dests, linked);
    }

    /// @brief Atomically increment this semaphore by `value` on the rectangle of cores.
    void inc_multicast(
        const Noc& noc,
        std::uint32_t noc_x_start,
        std::uint32_t noc_y_start,
        std::uint32_t noc_x_end,
        std::uint32_t noc_y_end,
        std::uint32_t value,
        std::uint32_t num_dests) {
        semaphore_detail::inc_multicast<SCOPE>(
            l1_offset_, noc, noc_x_start, noc_y_start, noc_x_end, noc_y_end, value, num_dests);
    }
#endif

private:
    std::uintptr_t l1_offset_;
};

template <std::uint32_t SEM_ID, SemScope TOK_SCOPE>
Semaphore(SemaphoreBindingToken<SEM_ID, TOK_SCOPE>) -> Semaphore<ProgrammableCoreType::TENSIX, TOK_SCOPE>;
