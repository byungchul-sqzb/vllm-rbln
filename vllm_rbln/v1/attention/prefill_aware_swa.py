# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Production PA-SWA contract helpers for RBLN attention.

Prefill-aware sliding-window attention (PA-SWA) is not equivalent to the
regular RBLN sliding-window path.  PA-SWA keeps all prefill tokens visible and
only slides the decode region:

    visible KV = all prefill tokens + latest ``sliding_window`` decode tokens

The helpers in this module are backend-neutral plumbing for status detection,
fail-closed guarding, and compact metadata construction.  They intentionally do
not mark the backend as ready; a future KV-cache manager and attention kernel
must consume this metadata before Unlimited-OCR PA-SWA decode can be claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class PrefillAwareSWAStatus:
    """Runtime contract for a PA-SWA-capable checkpoint/config.

    ``required`` means the model semantics need PA-SWA. ``backend_ready`` means
    the RBLN runtime has a PA-SWA-capable KV-cache/kernel implementation.
    ``active`` is derived and is true only when both are true.
    """

    required: bool
    backend_ready: bool
    active: bool
    sliding_window: int | None


def _call_bool_method(obj: Any, name: str) -> bool | None:
    method = getattr(obj, name, None)
    if method is None or not callable(method):
        return None
    return bool(method())


def _configured_sliding_window(model_or_config: Any) -> int | None:
    getter = getattr(model_or_config, "get_attention_sliding_window_size", None)
    if callable(getter):
        value = getter()
        if value is not None:
            return int(cast(Any, value))

    required_getter = getattr(
        model_or_config, "get_required_attention_sliding_window_size", None
    )
    if callable(required_getter):
        value = required_getter()
        if value is not None:
            return int(cast(Any, value))

    for attr in ("sliding_window_size", "sliding_window"):
        value = getattr(model_or_config, attr, None)
        if value is not None:
            return int(cast(Any, value))
    return None


def detect_prefill_aware_swa_status(
    model_or_config: Any,
) -> PrefillAwareSWAStatus:
    """Detect required/ready/active PA-SWA status from model or config markers."""

    required = bool(getattr(model_or_config, "requires_prefill_aware_swa", False))
    required_method = _call_bool_method(model_or_config, "requires_prefill_aware_swa")
    if required_method is not None:
        required = required_method

    backend_ready = bool(
        getattr(model_or_config, "rbln_prefill_aware_swa_backend_ready", False)
    )
    backend_ready_method = _call_bool_method(
        model_or_config, "is_prefill_aware_swa_backend_ready"
    )
    if backend_ready_method is not None:
        backend_ready = backend_ready_method

    # Backward-compatible active marker only.  It must not imply requirement.
    active_marker = getattr(model_or_config, "prefill_aware_swa", None)
    active_method = _call_bool_method(model_or_config, "is_prefill_aware_swa")
    active = required and backend_ready
    if active_method is not None:
        active = bool(active_method)
    elif active_marker is not None:
        active = bool(active_marker)
    active = active and required and backend_ready

    return PrefillAwareSWAStatus(
        required=required,
        backend_ready=backend_ready,
        active=active,
        sliding_window=_configured_sliding_window(model_or_config),
    )


def detect_prefill_aware_swa_status_from_vllm_config(
    vllm_config: Any,
) -> PrefillAwareSWAStatus:
    """Detect PA-SWA status from a vLLM config object."""

    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return PrefillAwareSWAStatus(
            required=False,
            backend_ready=False,
            active=False,
            sliding_window=None,
        )
    return detect_prefill_aware_swa_status(hf_config)


def regular_swa_requires_pa_swa_guard(
    status: PrefillAwareSWAStatus,
    *,
    sliding_window: int | None,
) -> bool:
    """Return whether regular SWA must be blocked for this PA-SWA status."""

    return status.required and not status.backend_ready and sliding_window is not None


def maybe_raise_for_incompatible_regular_swa(
    status: PrefillAwareSWAStatus,
    *,
    sliding_window: int | None,
    location: str,
) -> None:
    """Fail closed before PA-SWA-required models enter regular SWA."""

    if regular_swa_requires_pa_swa_guard(status, sliding_window=sliding_window):
        raise NotImplementedError(
            "Prefill-aware sliding-window attention is required by this model, "
            "but the RBLN PA-SWA KV-cache/kernel backend is not ready. "
            f"Refusing to enter the regular RBLN SWA path at {location} because "
            "regular SWA keeps only the latest window and would drop protected "
            "prefill tokens."
        )


def prefill_aware_visible_token_indices(
    *,
    seq_len: int,
    prefill_len: int,
    sliding_window: int,
) -> list[int]:
    """Return token positions visible to PA-SWA decode."""

    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if prefill_len < 0:
        raise ValueError(f"prefill_len must be non-negative, got {prefill_len}")
    if prefill_len > seq_len:
        raise ValueError(
            "prefill_len must be <= seq_len, got "
            f"prefill_len={prefill_len}, seq_len={seq_len}"
        )
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")

    decode_start = max(prefill_len, seq_len - sliding_window)
    return [*range(prefill_len), *range(decode_start, seq_len)]
