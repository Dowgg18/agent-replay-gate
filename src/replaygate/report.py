"""Two artifacts: JSON for machines, Markdown for the person triaging at 9am.

The Markdown one leads with the worst cases rather than the summary. A report
that opens with "mean score 87.3" gets skimmed; one that opens with the three
conversations that broke gets read.
"""

from __future__ import annotations

import json
from typing import Any

from .gate import GateOutcome
from .judge import Judge
from .rubric import CaseResult, Rubric


def to_json(
    results: list[CaseResult],
    rubric: Rubric,
    outcome: GateOutcome,
    judge: Judge | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    payload = {
        "meta": meta or {},
        "gate": outcome.as_dict(),
        "rubric": rubric.as_dict(),
        "judge": judge.summary() if judge else None,
        "cases": [
            {
                "id": r.case_id,
                "kind": r.kind,
                "score": r.score,
                "bucket": r.bucket,
                "defects": r.defects,
                "notes": r.notes,
                "error": r.error,
                "transcript": r.transcript,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def to_markdown(
    results: list[CaseResult],
    rubric: Rubric,
    outcome: GateOutcome,
    judge: Judge | None = None,
    meta: dict[str, Any] | None = None,
    worst: int = 5,
) -> str:
    lines: list[str] = []
    verdict = "PASS" if outcome.passed else "FAIL"
    lines.append(f"# Replay gate: {verdict}")
    lines.append("")

    if outcome.failures:
        lines.append("## Why it failed")
        lines.extend(f"- {f}" for f in outcome.failures)
        lines.append("")

    ranked = sorted(results, key=lambda r: (r.score, r.case_id))
    broken = [r for r in ranked if r.defects or r.error][:worst]
    if broken:
        lines.append(f"## Worst {len(broken)}")
        lines.append("")
        for r in broken:
            lines.append(f"### `{r.case_id}` — {r.score}/100 ({r.bucket})")
            if r.error:
                lines.append(f"- **crashed**: {r.error}")
            for note in r.notes:
                lines.append(f"- {note}")
            if r.transcript:
                lines.append("")
                lines.append("<details><summary>transcript</summary>")
                lines.append("")
                for turn in r.transcript:
                    lines.append(f"> **user**: {turn.get('user','')}")
                    lines.append(f"> **agent**: {turn.get('reply','')}")
                    node = turn.get("node")
                    if node:
                        lines.append(f"> _node_: `{node}`")
                    lines.append("")
                lines.append("</details>")
            lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- cases: **{outcome.total}**")
    lines.append(f"- mean score: **{outcome.mean_score:.1f}**")
    lines.append("")

    if outcome.bucket_counts:
        lines.append("| bucket | cases |")
        lines.append("|---|---|")
        for name, _floor in rubric.buckets:
            lines.append(f"| {name} | {outcome.bucket_counts.get(name, 0)} |")
        lines.append("")

    if outcome.signal_counts:
        lines.append("| signal | weight | cases |")
        lines.append("|---|---|---|")
        for signal, count in sorted(
            outcome.signal_counts.items(), key=lambda kv: -kv[1]
        ):
            weight = rubric.signals[signal].weight if signal in rubric.signals else "?"
            lines.append(f"| `{signal}` | {weight} | {count} |")
        lines.append("")

    if judge is not None:
        summary = judge.summary()
        lines.append("## Judge")
        lines.append("")
        if not summary["enabled"]:
            lines.append("- not configured — deterministic tier only")
        else:
            lines.append(f"- spent: {summary['spent']} / {summary['budget']}")
            if summary["budget_exhausted"]:
                lines.append("- **budget exhausted**: later cases were not judged")
            if summary["fail_opens"]:
                lines.append(
                    f"- **{summary['fail_opens']} fail-open(s)** — these cases were "
                    "assumed acceptable without a verdict:"
                )
                lines.extend(f"  - {r}" for r in summary["fail_open_reasons"])
        lines.append("")

    if meta:
        lines.append("## Run")
        lines.append("")
        lines.extend(f"- {k}: `{v}`" for k, v in meta.items())
        lines.append("")

    return "\n".join(lines)
