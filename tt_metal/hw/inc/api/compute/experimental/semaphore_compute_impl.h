// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "core_config.h"
#include "ckernel.h"

// Blackhole compute semaphore (SemScope::COMPUTE_ATOMIC): UNPACK <-> PACK synchronization on the Tensix
// hardware semaphore, Sync Unit index UNPACK_OPERAND_SYNC (3). The host rejects a ProgramSpec that binds
// such a semaphore from a DM kernel, so the only two agents that touch it are TRISC0 (UNPACK) and
// TRISC2 (PACK). This is the primitive the LLK itself uses for the MATH <-> PACK dest handshake.
//
// HARDWARE. The Sync Unit holds eight semaphores of {Value: 4 bits, Max: 4 bits}. SEMPOST/SEMGET
// atomically increment/decrement Value in one cycle; SEMPOST saturates at 15 and SEMGET floors at 0,
// both silently. SEMWAIT can block on Value == 0 or Value >= Max. RISC T0/T1/T2 read Value through a
// memory-mapped window (ckernel::semaphore_read); the read goes straight to the Sync Unit, is not
// cached in the RISC L0, and needs no fence.
//
//   up(n)           n x [ STALLWAIT(block SYNC, my engine idle) ; SEMPOST ]
//   down(n)         n x [ STALLWAIT(block SYNC, my engine idle) ; SEMGET  ]
//   wait_min(1)     SEMWAIT(block my engine, while Value == 0)          -- Tensix-side, RISC returns
//   wait_not_full() SEMWAIT(block my engine, while Value >= Max)        -- Tensix-side, RISC returns
//   wait_min(v)     tensix_sync ; RISC polls Value >= v   (v != 1: SEMWAIT has no such condition)
//   wait(v)         tensix_sync ; RISC polls Value == v
//   set(v)          STALLWAIT(block SYNC, my engine idle) ; SEMINIT(Max, Value = v)
//   value()         tensix_sync ; read Value
//
// MAX / BACK-PRESSURE. Max is the ring depth in credits, host-baked as COMPUTE_SEMAPHORE_MAX from
// SemaphoreAdvancedOptions::max_value (default: the hardware ceiling 15) and programmed by every SEMINIT
// here. With Max = depth, Value is "filled slots": the consumer blocks on empty (wait_min(1), SEMWAIT
// C0) and the producer blocks on full (wait_not_full(), SEMWAIT C1), both on the engine side, and the
// 15-credit saturation ceiling can never be reached. One 4-bit semaphore then does what a CB does with
// two counters.
//
// ORDERING (why up()/down() carry a STALLWAIT). SEMPOST/SEMGET execute in the Sync Unit, asynchronously
// from the packer/unpacker. The STALLWAIT holds the semaphore instruction at this thread's Wait Gate
// until this thread's engine is idle, so a consumer that observes the increment also observes the
// preceding packer writes (publish-after-data), and a producer that observes the decrement knows the
// unpacker has finished reading the slot (release-after-read). For a pure counter the wait passes
// immediately, so there is one primitive to reason about. This is the ManualTTSync.md "Tensix
// semaphores" pattern.
//
// WHY THE CONSUMER WAIT IS NOT A RISC POLL. up()/down() are issued to this thread's Tensix stream and
// retire only when their STALLWAIT clears, after the engine drains. A RISC read issued right after them
// sees the pre-update value. In the canonical loop
//     wait_min(1); <engine reads slot>; down(1);
// a RISC-polled wait would pass its next iteration on the credit it has just consumed but not yet
// retired: a double consume. SEMWAIT sits in the same in-order stream as this thread's own SEMGET, so it
// is evaluated after it, and it blocks only the engine instructions that follow (UNPACR / PACR); the
// RISC keeps issuing. SEMWAIT offers only "Value == 0" and "Value >= Max", so wait(v) and
// wait_min(v != 1) fall back to a RISC poll preceded by tensix_sync(), which retires this thread's own
// posted SEMPOST/SEMGET first. value() takes the same sync so it reports a settled number.
//
// CONTRACT. wait()/wait_min() order this thread's subsequent Tensix engine instructions after the
// condition. They do not order RISC-side L1 loads issued by the kernel itself; a kernel that reads the
// buffer with the RISC must poll value() instead.
//
// INITIAL VALUE. Nothing zeroes the Sync Unit between programs and the host cannot write it, so the
// semaphore is seeded by compute_kernel_hw_startup() (2.0), which calls compute_semaphore_hw_startup()
// on PACK. That is race-free without a cross-thread handshake because PACK is the producer (its first
// SEMPOST is queued behind the SEMINIT) and a balanced previous kernel left Value at 0, so UNPACK reads
// 0 whether its first wait runs before or after the SEMINIT: the invariant llk_math_pack_sync_init
// relies on for MATH_PACK. Consequently a compute-bound semaphore must have initial_value 0
// (host-enforced) and every kernel must leave it balanced at 0.
//
// LIMITS. Value is 0..15; more than 15 outstanding credits lose posts silently, so a producer that does
// not use wait_not_full() must be bounded to at most 15 ahead of its consumer by other means. The
// semaphore is core-local (not a NoC atomic, not reachable from a DM core). Index 3 is the only free
// Sync Unit semaphore on Blackhole, so a program may bind exactly one compute semaphore
// (host-enforced); the bound id is not used as an address.

// The compute semaphore's Max: host-baked from SemaphoreAdvancedOptions::max_value when set, else the
// hardware ceiling. Every SEMINIT in this file programs it.
#ifndef COMPUTE_SEMAPHORE_MAX
#define COMPUTE_SEMAPHORE_MAX 15
#endif
inline constexpr std::uint32_t kComputeSemaphoreMax = COMPUTE_SEMAPHORE_MAX;
static_assert(kComputeSemaphoreMax >= 1 && kComputeSemaphoreMax <= 15, "COMPUTE_SEMAPHORE_MAX must be 1..15");

/**
 * @brief Seed the compute semaphore for this kernel. Called by compute_kernel_hw_startup() (2.0) on the
 * PACK thread; kernels do not call it directly.
 *
 * Sets the Tensix hardware semaphore backing SemScope::COMPUTE_ATOMIC to Value 0 and Max =
 * COMPUTE_SEMAPHORE_MAX. Must run on the producing thread (PACK), whose first up() is then queued behind
 * it; requires that the previous kernel left the semaphore balanced at 0, which is why the host rejects
 * a nonzero initial_value on a compute binding. No-op on non-Blackhole builds and on the other threads.
 */
__attribute__((always_inline)) inline void compute_semaphore_hw_startup() {
#if defined(ARCH_BLACKHOLE) && defined(TRISC_PACK)
    ckernel::t6_semaphore_init(ckernel::semaphore::UNPACK_OPERAND_SYNC, /*value=*/0, kComputeSemaphoreMax);
#endif
}

namespace semaphore_detail {

// Dependent false, so a static_assert in a class-template member fires only on instantiation.
template <SemScope>
inline constexpr bool always_false = false;

// The bound id selects nothing here: every compute semaphore is the one free Sync Unit index. The
// function keeps the name sem_l1_offset so the shared Semaphore class template (semaphore.h) needs no
// change; the returned value is the hardware index, threaded through up/down/wait/... below.
template <ProgrammableCoreType core_type, SemScope scope>
__attribute__((always_inline)) inline std::uintptr_t sem_l1_offset(std::uint32_t /*id*/) {
    static_assert(core_type == ProgrammableCoreType::TENSIX, "compute semaphores require a Tensix core");
    static_assert(scope == SemScope::COMPUTE_ATOMIC, "Blackhole compute supports COMPUTE_ATOMIC only");
    return ckernel::semaphore::UNPACK_OPERAND_SYNC;
}

#if defined(ARCH_BLACKHOLE) && (defined(TRISC_UNPACK) || defined(TRISC_PACK))

// STALLWAIT condition naming this thread's own engine (so up()/down() are ordered after its work), and
// the matching SEMWAIT block bit (so a wait holds exactly this thread's engine instructions).
#if defined(TRISC_UNPACK)
inline constexpr std::uint32_t kEngineIdle = ckernel::p_stall::UNPACK;  // C1|C2: both unpackers idle
inline constexpr std::uint32_t kEngineBlock = ckernel::p_stall::STALL_UNPACK;  // B3: block UNPACR
#else
inline constexpr std::uint32_t kEngineIdle = ckernel::p_stall::PACK;  // C3: packer idle
inline constexpr std::uint32_t kEngineBlock = ckernel::p_stall::STALL_PACK;  // B2: block PACR
#endif

template <SemScope scope>
__attribute__((always_inline)) inline std::uint32_t load(std::uintptr_t index) {
    static_assert(scope == SemScope::COMPUTE_ATOMIC);
    return ckernel::semaphore_read(static_cast<std::uint8_t>(index));
}

// Settled value: retire this thread's own posted SEMPOST/SEMGET (and the engine work their STALLWAITs
// wait on) before reading.
template <SemScope scope>
__attribute__((always_inline)) inline std::uint32_t current(std::uintptr_t index) {
    ckernel::tensix_sync();
    return load<scope>(index);
}

template <ProgrammableCoreType core_type, SemScope scope>
__attribute__((always_inline)) inline void up(std::uintptr_t index, std::uint32_t value) {
    static_assert(scope == SemScope::COMPUTE_ATOMIC);
    const std::uint8_t idx = static_cast<std::uint8_t>(index);
    for (std::uint32_t i = 0; i < value; ++i) {
        ckernel::t6_semaphore_post<kEngineIdle>(idx);  // STALLWAIT(my engine) + SEMPOST
    }
}

template <ProgrammableCoreType core_type, SemScope scope>
__attribute__((always_inline)) inline void down(std::uintptr_t index, std::uint32_t value) {
    static_assert(scope == SemScope::COMPUTE_ATOMIC);
    const std::uint8_t idx = static_cast<std::uint8_t>(index);
    for (std::uint32_t i = 0; i < value; ++i) {
        ckernel::t6_semaphore_get<kEngineIdle>(idx);  // STALLWAIT(my engine) + SEMGET
    }
}

template <SemScope scope>
__attribute__((always_inline)) inline void wait(std::uintptr_t index, std::uint32_t value) {
    static_assert(scope == SemScope::COMPUTE_ATOMIC);
    // No SEMWAIT condition for "== v": RISC poll, after retiring this thread's own posted ops.
    WAYPOINT("NSW");
    ckernel::tensix_sync();
    while (load<scope>(index) != value) {
    }
    WAYPOINT("NSD");
}

template <SemScope scope>
__attribute__((always_inline)) inline void wait_min(std::uintptr_t index, std::uint32_t value) {
    static_assert(scope == SemScope::COMPUTE_ATOMIC);
    if (value == 1) {
        // Tensix-side: hold this thread's engine instructions while Value == 0. In order with this
        // thread's own SEMGETs, so a just-consumed credit is never counted again. The RISC returns.
        ckernel::t6_semaphore_wait_on_zero<kEngineBlock>(static_cast<std::uint8_t>(index));
        return;
    }
    WAYPOINT("NSMW");
    ckernel::tensix_sync();
    while (load<scope>(index) < value) {
    }
    WAYPOINT("NSMD");
}

// Tensix-side: hold this thread's engine instructions while Value >= Max (the ring is full). In order
// with this thread's own SEMPOSTs, so a credit this thread has posted but not yet retired still counts.
template <SemScope scope>
__attribute__((always_inline)) inline void wait_not_full(std::uintptr_t index) {
    static_assert(scope == SemScope::COMPUTE_ATOMIC);
    ckernel::t6_semaphore_wait_on_max<kEngineBlock>(static_cast<std::uint8_t>(index));
}

template <SemScope scope>
__attribute__((always_inline)) inline void set(std::uintptr_t index, std::uint32_t value) {
    static_assert(scope == SemScope::COMPUTE_ATOMIC);
    // Absolute assignment via SEMINIT (Max re-programmed to the host-baked capacity), ordered after this
    // thread's engine work for the same reason as up()/down().
    TTI_STALLWAIT(ckernel::p_stall::STALL_SYNC, kEngineIdle);
    ckernel::t6_semaphore_init(
        static_cast<std::uint8_t>(index), static_cast<std::uint8_t>(value), kComputeSemaphoreMax);
}

#endif  // ARCH_BLACKHOLE && (UNPACK || PACK)

}  // namespace semaphore_detail
