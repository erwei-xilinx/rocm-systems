// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
class Instruction;

/// @brief Internal wait counters tracked by waitcheck.
///
/// @details GFX12 targets use the split counter names directly. Legacy gfx9
/// style targets reuse Load for vmcnt, Ds for lgkmcnt, and Exp for expcnt.
enum class WaitCounterKind : uint8_t {
  Load = 0,
  Store,
  Ds,
  Km,
  Sample,
  Bvh,
  Exp,
  X,
  Async,
  Tensor,
  VmVsrc,
  VaVdst,
  Depctr,
  Count,
};

[[nodiscard]] std::string_view wait_counter_name(WaitCounterKind counter);

namespace waitcheck_detail {
inline constexpr size_t kCounterCount = static_cast<size_t>(WaitCounterKind::Count);

[[nodiscard]] inline bool is_supported_waitcheck_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

[[nodiscard]] inline uint32_t default_wavefront_size(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                 arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5
             ? 32
             : 64;
}

[[nodiscard]] inline bool starts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] inline bool equals_ignore_ascii_case(std::string_view lhs, std::string_view rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const auto to_lower = [](char c) {
      return c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : c;
    };
    if (to_lower(lhs[i]) != to_lower(rhs[i]))
      return false;
  }
  return true;
}

enum class WaitcntModel { LegacyNoVscnt, LegacyVscnt, SplitGfx12 };

[[nodiscard]] inline WaitcntModel waitcnt_model(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return WaitcntModel::LegacyNoVscnt;
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return WaitcntModel::LegacyVscnt;
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_CDNA5:
    return WaitcntModel::SplitGfx12;
  default:
    throw std::invalid_argument("unsupported waitcheck architecture");
  }
}

[[nodiscard]] inline bool uses_legacy_waitcnt(rj_code_arch_t arch) {
  return waitcnt_model(arch) != WaitcntModel::SplitGfx12;
}

[[nodiscard]] inline bool has_legacy_vscnt(rj_code_arch_t arch) {
  return waitcnt_model(arch) == WaitcntModel::LegacyVscnt;
}

[[nodiscard]] inline bool supports_expert_scheduling(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

[[nodiscard]] inline bool tracks_committed_vgpr_generations(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] inline bool tracks_gfx12_sgpr_hazards(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] inline WaitCounterKind smem_wait_counter(rj_code_arch_t arch) {
  return uses_legacy_waitcnt(arch) ? WaitCounterKind::Ds : WaitCounterKind::Km;
}

[[nodiscard]] inline WaitCounterKind vmem_store_wait_counter(rj_code_arch_t arch) {
  return waitcnt_model(arch) == WaitcntModel::LegacyNoVscnt ? WaitCounterKind::Load
                                                            : WaitCounterKind::Store;
}

[[nodiscard]] inline WaitCounterKind image_sample_wait_counter(rj_code_arch_t arch) {
  return uses_legacy_waitcnt(arch) ? WaitCounterKind::Load : WaitCounterKind::Sample;
}

[[nodiscard]] inline WaitCounterKind image_bvh_wait_counter(rj_code_arch_t arch) {
  return uses_legacy_waitcnt(arch) ? WaitCounterKind::Load : WaitCounterKind::Bvh;
}

struct LegacyWaitcnt {
  uint32_t vmcnt = 0;
  uint32_t expcnt = 0;
  uint32_t lgkmcnt = 0;
};

[[nodiscard]] inline LegacyWaitcnt decode_legacy_waitcnt(uint32_t value) {
  return {
      (value & 0xFu) | (((value >> 14u) & 0x3u) << 4u),
      (value >> 4u) & 0x7u,
      (value >> 8u) & 0xFu,
  };
}

[[nodiscard]] inline LegacyWaitcnt decode_gfx11_waitcnt(uint32_t value) {
  return {
      (value >> 10u) & 0x3Fu,
      value & 0x7u,
      (value >> 4u) & 0x3Fu,
  };
}

[[nodiscard]] inline LegacyWaitcnt decode_legacy_waitcnt(uint32_t value, rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5
             ? decode_gfx11_waitcnt(value)
             : decode_legacy_waitcnt(value);
}

[[nodiscard]] inline std::string wait_expression(WaitCounterKind counter, uint32_t required_count,
                                                 rj_code_arch_t arch) {
  std::ostringstream os;
  if (uses_legacy_waitcnt(arch)) {
    switch (counter) {
    case WaitCounterKind::Load:
      os << "s_waitcnt vmcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Ds:
      os << "s_waitcnt lgkmcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Exp:
      os << "s_waitcnt expcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Store:
      if (has_legacy_vscnt(arch)) {
        os << "s_waitcnt_vscnt null, " << required_count;
        return os.str();
      }
      break;
    default:
      break;
    }
  }
  if (counter == WaitCounterKind::X) {
    os << "s_wait_xcnt " << required_count;
  } else if (counter == WaitCounterKind::VmVsrc || counter == WaitCounterKind::VaVdst) {
    os << (has_legacy_vscnt(arch) ? "s_waitcnt_depctr " : "s_wait_alu ")
       << (counter == WaitCounterKind::VmVsrc ? "depctr_vm_vsrc(" : "depctr_va_vdst(")
       << required_count << ")";
  } else {
    os << "s_wait_" << wait_counter_name(counter) << " <= " << required_count;
  }
  return os.str();
}

enum class WaitEventKind {
  Unknown,
  VmemNoSamplerLoad,
  FlatLoad,
  VmemStore,
  FlatStore,
  Ds,
  Gds,
  Smem,
  Sample,
  Bvh,
  Export,
  SccWrite,
  SqMessage,
  GlobalInv,
  GlobalWb,
  LdsDirect,
  AsyncLdsLoad,
  AsyncLdsStore,
  AsyncBarrier,
  TensorLdsLoad,
  TensorLdsStore,
  Count,
};

inline constexpr size_t kWaitEventKindCount = static_cast<size_t>(WaitEventKind::Count);
inline constexpr uint8_t kNoPendingEventAge = std::numeric_limits<uint8_t>::max();

enum class TrackedRegisterSource {
  None,
  Defs,
  Uses,
  VectorUses,
  StoreDataUses,
};

struct ClassifiedEvent {
  ClassifiedEvent(WaitCounterKind counter = WaitCounterKind::Load,
                  WaitEventKind kind = WaitEventKind::Unknown,
                  TrackedRegisterSource registers = TrackedRegisterSource::Defs,
                  bool check_uses = true, bool check_defs = true, bool check_exec_defs = false,
                  std::optional<RegisterRef> special_reg = std::nullopt,
                  std::optional<int64_t> barrier_id = std::nullopt, bool check_memory_order = false,
                  bool check_program_end = false, bool check_counter_parity_order = false)
      : counter(counter), kind(kind), registers(registers), check_uses(check_uses),
        check_defs(check_defs), check_exec_defs(check_exec_defs), special_reg(special_reg),
        barrier_id(barrier_id), check_memory_order(check_memory_order),
        check_program_end(check_program_end),
        check_counter_parity_order(check_counter_parity_order) {}

  WaitCounterKind counter = WaitCounterKind::Load;
  WaitEventKind kind = WaitEventKind::Unknown;
  TrackedRegisterSource registers = TrackedRegisterSource::Defs;
  bool check_uses = true;
  bool check_defs = true;
  bool check_exec_defs = false;
  std::optional<RegisterRef> special_reg;
  std::optional<int64_t> barrier_id;
  bool check_memory_order = false;
  bool check_program_end = false;
  bool check_counter_parity_order = false;
};

// ISA classification and wait-field decoding have no pending-state or CFG
// dependencies. The checker composes these policies with its dataflow state.
struct WaitcheckTarget {
  [[nodiscard]] static size_t counter_index(WaitCounterKind counter);

  [[nodiscard]] static uint32_t maximum_dependency_wait(rj_code_arch_t arch,
                                                        WaitCounterKind counter);

  [[nodiscard]] static std::optional<uint32_t> counter_no_wait_value(rj_code_arch_t arch,
                                                                     WaitCounterKind counter);

  [[nodiscard]] static std::optional<std::array<std::optional<uint32_t>, kCounterCount>>
  explicit_wait_fields(const Instruction &inst, rj_code_arch_t arch);

  [[nodiscard]] static std::optional<std::array<std::optional<uint32_t>, kCounterCount>>
  embedded_wait_fields(const Instruction &inst, rj_code_arch_t arch);

  [[nodiscard]] static bool is_xcnt_vmem_kind(WaitEventKind kind);

  [[nodiscard]] static uint32_t depctr_field(uint32_t value, uint32_t shift, uint32_t width);

  [[nodiscard]] static std::optional<int64_t> first_barrier_id(const Instruction &inst);

  [[nodiscard]] static bool is_cdna4_mubuf_lds_load(const Instruction &inst, rj_code_arch_t arch);

  [[nodiscard]] static std::optional<uint32_t> vinterp_wait_exp(const Instruction &inst);

  [[nodiscard]] static bool is_dsdir(std::string_view mnemonic);

  [[nodiscard]] static std::optional<uint32_t> dsdir_wait_va_vdst(const Instruction &inst);

  [[nodiscard]] static std::optional<uint32_t> dsdir_wait_vm_vsrc(const Instruction &inst);

  [[nodiscard]] static bool is_scalar_memory_op(std::string_view mnemonic);

  [[nodiscard]] static bool is_vmem_store(std::string_view mnemonic);

  [[nodiscard]] static bool is_vmem_atomic(std::string_view mnemonic);

  [[nodiscard]] static bool is_image_atomic(std::string_view mnemonic);

  [[nodiscard]] static bool is_gfx1250_smem(std::string_view mnemonic);

  [[nodiscard]] static bool uses_ds_wait_counter(std::string_view mnemonic);

  [[nodiscard]] static bool is_async_lds_load(std::string_view mnemonic);

  [[nodiscard]] static bool is_async_lds_store(std::string_view mnemonic);

  [[nodiscard]] static bool is_tensor_lds_load(std::string_view mnemonic);

  [[nodiscard]] static bool is_tensor_lds_store(std::string_view mnemonic);

  [[nodiscard]] static std::vector<ClassifiedEvent> classify_events(const Instruction &inst,
                                                                    rj_code_arch_t arch);
};

} // namespace waitcheck_detail
} // namespace rocjitsu
