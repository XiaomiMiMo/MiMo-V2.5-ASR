"""
memory_guard.py — System memory detection and conservative threshold enforcement.

On Apple Silicon (MPS), RAM is shared with GPU (unified memory architecture),
so total system RAM is the true limit for model loading.

MiMo-V2.5-ASR memory profile:
  - Main LLM model (bfloat16): ~14 GB
  - Audio tokenizer (bfloat16):  ~1 GB
  - KV cache + activations:      ~3-5 GB
  - Audio preprocessing buffers: ~1 GB
  - OS + Python + other:         ~2 GB
  ─────────────────────────────────────
  Minimum recommended:           ~22 GB (free before load)
  Comfortable headroom:          ~28 GB (recommended free)
"""

import platform
import subprocess
import psutil
import torch


# ── Constants ──────────────────────────────────────────────────────────────
# Estimated peak memory required to safely load and run the model (bytes)
MODEL_MIN_REQUIRED_GB = 22.0   # absolute floor — will likely OOM below this
MODEL_RECOMMENDED_GB  = 28.0   # comfortable headroom for stable operation
CONSERVATIVE_FREE_RATIO = 0.30  # keep at least 30% of total RAM free after load


def get_system_memory_info() -> dict:
    """
    Return a dict with detailed memory stats (all values in bytes unless noted).

    Keys:
        total_bytes         – total physical RAM
        available_bytes     – OS-reported available (not yet used / easily reclaimable)
        used_bytes          – memory in active use
        percent_used        – float 0-100
        total_gb / available_gb / used_gb – human-readable floats
        is_apple_silicon    – bool
        unified_memory      – bool (True on Apple Silicon: GPU shares this pool)
        swap_total_bytes / swap_used_bytes / swap_free_bytes
    """
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    is_apple = platform.system() == "Darwin" and platform.machine() == "arm64"

    return {
        "total_bytes":      vm.total,
        "available_bytes":  vm.available,
        "used_bytes":       vm.used,
        "percent_used":     vm.percent,
        "total_gb":         vm.total      / 1024**3,
        "available_gb":     vm.available  / 1024**3,
        "used_gb":          vm.used       / 1024**3,
        "is_apple_silicon": is_apple,
        "unified_memory":   is_apple,   # Apple Silicon = unified RAM/VRAM
        "swap_total_bytes": swap.total,
        "swap_used_bytes":  swap.used,
        "swap_free_bytes":  swap.free,
        "swap_total_gb":    swap.total / 1024**3,
        "swap_used_gb":     swap.used  / 1024**3,
    }


def get_process_memory_gb() -> float:
    """Return current process RSS memory usage in GB."""
    import os
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024**3


def get_gpu_memory_info() -> dict:
    """
    Return GPU memory info. On Apple Silicon, returns system memory stats
    (unified architecture). On CUDA, returns device memory.
    """
    result = {"backend": "none", "total_gb": 0.0, "free_gb": 0.0, "used_gb": 0.0}

    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        reserved = torch.cuda.memory_reserved(dev)
        allocated = torch.cuda.memory_allocated(dev)
        free = total - reserved
        result.update({
            "backend":  "cuda",
            "total_gb": total     / 1024**3,
            "free_gb":  free      / 1024**3,
            "used_gb":  allocated / 1024**3,
        })
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS shares system RAM — report system available as GPU free
        vm = psutil.virtual_memory()
        result.update({
            "backend":  "mps (unified)",
            "total_gb": vm.total     / 1024**3,
            "free_gb":  vm.available / 1024**3,
            "used_gb":  vm.used      / 1024**3,
        })

    return result


# ── Threshold checks ────────────────────────────────────────────────────────

class MemoryCheckResult:
    """Result of a memory threshold check."""

    def __init__(
        self,
        ok: bool,
        level: str,          # "ok" | "warning" | "critical"
        available_gb: float,
        total_gb: float,
        required_gb: float,
        message: str,
        details: str = "",
    ):
        self.ok           = ok
        self.level        = level
        self.available_gb = available_gb
        self.total_gb     = total_gb
        self.required_gb  = required_gb
        self.message      = message
        self.details      = details

    def __bool__(self):
        return self.ok

    def format_report(self) -> str:
        icon = {"ok": "✅", "warning": "⚠️", "critical": "❌"}.get(self.level, "ℹ️")
        lines = [
            f"{icon} {self.message}",
            f"   可用記憶體: {self.available_gb:.1f} GB / 總計: {self.total_gb:.1f} GB",
            f"   模型需求:   ≥ {self.required_gb:.1f} GB (建議 ≥ {MODEL_RECOMMENDED_GB:.0f} GB)",
        ]
        if self.details:
            lines.append(f"   {self.details}")
        return "\n".join(lines)


def check_memory_before_load(
    required_gb: float = MODEL_MIN_REQUIRED_GB,
    recommended_gb: float = MODEL_RECOMMENDED_GB,
) -> MemoryCheckResult:
    """
    Check if there is sufficient memory to safely load the model.

    Returns a MemoryCheckResult with:
      - ok=True  if available memory >= required_gb  (may show warning if < recommended)
      - ok=False if available memory < required_gb   (critical — loading may OOM)
    """
    info = get_system_memory_info()
    avail_gb = info["available_gb"]
    total_gb = info["total_gb"]
    swap_used_gb = info["swap_used_gb"]

    # Extra detail about unified memory
    arch_note = ""
    if info["unified_memory"]:
        arch_note = "Apple Silicon 統一記憶體（RAM 與 GPU 共用）"

    # ── Critical: definitely not enough ─────────────────────────────────
    if avail_gb < required_gb:
        shortage = required_gb - avail_gb
        details = f"記憶體不足 {shortage:.1f} GB。{arch_note}"
        if swap_used_gb > 2.0:
            details += f" Swap 已使用 {swap_used_gb:.1f} GB，系統已在大量使用虛擬記憶體。"
        return MemoryCheckResult(
            ok=False,
            level="critical",
            available_gb=avail_gb,
            total_gb=total_gb,
            required_gb=required_gb,
            message=(
                f"記憶體不足：可用 {avail_gb:.1f} GB，"
                f"需要至少 {required_gb:.0f} GB 才能安全載入模型"
            ),
            details=details,
        )

    # ── Warning: technically enough but tight ───────────────────────────
    if avail_gb < recommended_gb:
        details = f"建議至少 {recommended_gb:.0f} GB 可用，以確保穩定運行。{arch_note}"
        if swap_used_gb > 1.0:
            details += f" Swap 已使用 {swap_used_gb:.1f} GB，記憶體壓力較高。"
        return MemoryCheckResult(
            ok=True,   # allow loading but show warning
            level="warning",
            available_gb=avail_gb,
            total_gb=total_gb,
            required_gb=required_gb,
            message=(
                f"記憶體偏低：可用 {avail_gb:.1f} GB "
                f"（建議 ≥ {recommended_gb:.0f} GB），可能出現效能下降"
            ),
            details=details,
        )

    # ── OK ───────────────────────────────────────────────────────────────
    after_load_estimate = avail_gb - required_gb
    details = f"載入後預估剩餘: ~{after_load_estimate:.1f} GB。{arch_note}"
    return MemoryCheckResult(
        ok=True,
        level="ok",
        available_gb=avail_gb,
        total_gb=total_gb,
        required_gb=required_gb,
        message=f"記憶體充足：可用 {avail_gb:.1f} GB / 總計 {total_gb:.1f} GB",
        details=details,
    )


def format_memory_status() -> str:
    """Return a concise multi-line memory status string for display in the UI."""
    info = get_system_memory_info()
    gpu  = get_gpu_memory_info()
    proc_gb = get_process_memory_gb()

    lines = []

    # System memory bar
    used_pct = info["percent_used"]
    bar_len = 20
    filled  = int(bar_len * used_pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    lines.append(f"系統記憶體: [{bar}] {used_pct:.0f}%")
    lines.append(
        f"  總計: {info['total_gb']:.1f} GB  "
        f"已用: {info['used_gb']:.1f} GB  "
        f"可用: {info['available_gb']:.1f} GB"
    )

    # Swap
    if info["swap_total_gb"] > 0:
        lines.append(
            f"  Swap: {info['swap_used_gb']:.1f} / {info['swap_total_gb']:.1f} GB"
            + (" ⚠️ 高" if info["swap_used_gb"] > info["swap_total_gb"] * 0.5 else "")
        )

    # GPU / Unified
    if gpu["backend"] != "none":
        if gpu["backend"] == "mps (unified)":
            lines.append(f"  GPU: Apple Silicon 統一記憶體（同上）")
        else:
            lines.append(
                f"  GPU ({gpu['backend']}): "
                f"已用 {gpu['used_gb']:.1f} GB / 可用 {gpu['free_gb']:.1f} GB"
            )

    # Process self
    lines.append(f"  本程序佔用: {proc_gb:.1f} GB")

    # Architecture note
    if info["is_apple_silicon"]:
        lines.append("  📱 Apple Silicon M 系列（RAM = GPU 共用統一記憶體）")

    # Threshold check
    check = check_memory_before_load()
    lines.append("")
    lines.append(check.format_report())

    return "\n".join(lines)
