"""Runtime metrics helpers for Unlimited-OCR experiments."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComponentMetric:
    """One measured component execution."""

    component: str
    backend: str
    elapsed_sec: float
    detail: str = ""
    calls: int = 1


@dataclass
class StageMetric:
    """A user-facing grouped timing row."""

    stage: str
    backend: str
    elapsed_sec: float
    calls: int
    detail: str = ""


@dataclass
class RuntimeMetrics:
    """Collect component timings for one CLI run."""

    rows: list[ComponentMetric] = field(default_factory=list)

    @contextmanager
    def measure(
        self, component: str, backend: str = "cpu", detail: str = ""
    ) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(component, backend, time.perf_counter() - started, detail)

    def add(
        self, component: str, backend: str, elapsed_sec: float, detail: str = ""
    ) -> None:
        self.rows.append(
            ComponentMetric(
                component=component,
                backend=backend,
                elapsed_sec=elapsed_sec,
                detail=detail,
            )
        )

    def grouped_rows(self) -> list[ComponentMetric]:
        grouped: dict[tuple[str, str, str], ComponentMetric] = {}
        order: list[tuple[str, str, str]] = []
        for row in self.rows:
            key = (row.component, row.backend, row.detail)
            if key not in grouped:
                grouped[key] = ComponentMetric(
                    component=row.component,
                    backend=row.backend,
                    elapsed_sec=row.elapsed_sec,
                    detail=row.detail,
                    calls=row.calls,
                )
                order.append(key)
            else:
                grouped[key].elapsed_sec += row.elapsed_sec
                grouped[key].calls += row.calls
        return [grouped[key] for key in order]

    def stage_rows(self) -> list[StageMetric]:
        grouped: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in self.grouped_rows():
            stage = _stage_for(row.component, row.detail)
            if stage not in grouped:
                grouped[stage] = {
                    "elapsed_sec": 0.0,
                    "calls": 0,
                    "backends": set(),
                    "components": [],
                }
                order.append(stage)
            grouped[stage]["elapsed_sec"] += row.elapsed_sec
            grouped[stage]["calls"] = max(grouped[stage]["calls"], row.calls)
            grouped[stage]["backends"].add(row.backend)
            grouped[stage]["components"].append(_component_label(row))

        return [
            StageMetric(
                stage=stage,
                backend=_backend_label(grouped[stage]["backends"]),
                elapsed_sec=grouped[stage]["elapsed_sec"],
                calls=grouped[stage]["calls"],
                detail=", ".join(grouped[stage]["components"]),
            )
            for stage in order
        ]

    def to_list(self) -> list[dict[str, Any]]:
        return [
            {
                "component": row.component,
                "backend": row.backend,
                "elapsed_sec": row.elapsed_sec,
                "calls": row.calls,
                "detail": row.detail,
            }
            for row in self.grouped_rows()
        ]

    def stages_to_list(self) -> list[dict[str, Any]]:
        return [
            {
                "stage": row.stage,
                "backend": row.backend,
                "elapsed_sec": row.elapsed_sec,
                "calls": row.calls,
                "detail": row.detail,
            }
            for row in self.stage_rows()
        ]

    def render_table(self) -> str:
        return _render_component_table(self.grouped_rows())

    def render_stage_table(self) -> str:
        rows = self.stage_rows()
        if not rows:
            return "Stage timings: no measured stages"

        headers = ["stage", "backend", "calls", "time_sec", "detail"]
        table_rows = [
            [
                row.stage,
                row.backend,
                str(row.calls),
                f"{row.elapsed_sec:.6f}",
                row.detail,
            ]
            for row in rows
        ]
        return _render_table("Stage timings", headers, table_rows)


def _render_component_table(rows: list[ComponentMetric]) -> str:
    if not rows:
        return "Component timings: no measured components"

    headers = ["component", "backend", "calls", "time_sec", "detail"]
    table_rows = [
        [
            row.component,
            row.backend,
            str(row.calls),
            f"{row.elapsed_sec:.6f}",
            row.detail,
        ]
        for row in rows
    ]
    return _render_table("Component timings", headers, table_rows)


def _render_table(title: str, headers: list[str], table_rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in table_rows))
        for idx in range(len(headers))
    ]

    def fmt_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    sep = "-+-".join("-" * width for width in widths)
    lines = [title, fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in table_rows)
    return "\n".join(lines)


def _stage_for(component: str, detail: str) -> str:
    if component in {"model_load", "preprocess"}:
        return component
    if component in {"sam_model", "vision_model", "projector"}:
        return component
    if component == "embed_tokens" and detail == "prompt":
        return "multimodal_embed"
    if component in {"hf_generate", "token_select"}:
        return "language_decode"
    if component.startswith("language_prefill_"):
        return "language_prefill"
    if component.startswith("language_decode_compact_"):
        return component
    if component.startswith("language_decode_"):
        return "language_decode"
    return "other"


def _backend_label(backends: set[str]) -> str:
    if len(backends) == 1:
        return next(iter(backends))
    if "rbln" in backends and "cpu" in backends:
        return "mixed"
    return "+".join(sorted(backends))


def _component_label(row: ComponentMetric) -> str:
    return row.component if not row.detail else f"{row.component}:{row.detail}"
