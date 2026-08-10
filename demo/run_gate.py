"""Wire the demo agent to the harness and run the gate.

    python -m demo.run_gate            # deterministic tier
    python -m demo.run_gate --judged   # adds the judged tier (scripted backend)

Exit code is the point: 0 or 1, nothing to parse.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from replaygate import (
    Count,
    Fake,
    Forbid,
    Judge,
    ReplayConfig,
    Thresholds,
    evaluate,
    load_cases,
    replay_all,
    seam,
    to_json,
    to_markdown,
)
from replaygate.judge import ScriptedJudge
from replaygate.rubric import DEFAULT_RUBRIC

from . import deps, fixtures
from .agent import BookingAgent

HERE = pathlib.Path(__file__).parent
REPORTS = HERE.parent / "reports"


def build_agent() -> BookingAgent:
    return BookingAgent()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the demo replay gate")
    parser.add_argument(
        "--judged",
        action="store_true",
        help="also run the judged tier with a scripted backend",
    )
    parser.add_argument(
        "--strict-determinism",
        action="store_true",
        help="FORBID the model seam instead of counting it: any fallback is a hard failure",
    )
    args = parser.parse_args(argv)

    cases = load_cases(str(HERE / "cases.yaml"))

    # The store and counter are rebuilt per run so cases cannot leak into each
    # other through the fixture.
    deps.store = fixtures.build_store(read_only=True)
    deps.counter = fixtures.build_counter()
    deps.llm = _ScriptedModel()

    # The seam is on the *method*, not the object: a policy can only observe a
    # crossing it is able to wrap, and wrapping the object would leave every call
    # on it invisible.
    model_seam = (
        seam(
            "demo.deps.llm.complete",
            Forbid("the deterministic path must not call the model"),
        )
        if args.strict_determinism
        else seam("demo.deps.llm.complete", Count(deps.llm.complete, label="model"))
    )

    config = ReplayConfig(
        rubric=DEFAULT_RUBRIC,
        seams=[
            model_seam,
            seam("demo.deps.store", Fake(deps.store)),
            seam("demo.deps.counter", Fake(deps.counter)),
        ],
        count_signal_map={"model": "model_fallback"},
        # Escalation cases feed the agent deliberate nonsense; reaching for the
        # model there is the designed behaviour, not a defect.
        count_exempt_kinds={"model_fallback": ("escalation",)},
        judge=Judge(backend=ScriptedJudge(), budget=0.05) if args.judged else None,
        judge_kinds=("booking",) if args.judged else (),
    )

    results = replay_all(cases, build_agent, config)

    # The flag has to move the gate, not just the seam. A --strict run that still
    # passes with two fallbacks is a flag that lies about what it enforces.
    thresholds = Thresholds(
        min_mean_score=85.0,
        max_critical_rate=0.0,
        max_signal_rates={"model_fallback": 0.0 if args.strict_determinism else 0.35},
    )
    outcome = evaluate(results, config.rubric, thresholds)

    meta = {
        "suite": "demo",
        "cases": len(cases),
        "tier": "judged" if args.judged else "deterministic",
        "strict_determinism": args.strict_determinism,
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "report.json").write_text(
        to_json(results, config.rubric, outcome, config.judge, meta), encoding="utf-8"
    )
    markdown = to_markdown(results, config.rubric, outcome, config.judge, meta)
    (REPORTS / "report.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nreports written to {REPORTS}")
    return outcome.exit_code


class _ScriptedModel:
    """Stands in for the LLM on the counted tier. Deterministic on purpose: the
    point of this tier is reproducibility, not model quality."""

    def complete(self, prompt: str, **_: object) -> str:  # noqa: ARG002
        return "Could you rephrase that?"


if __name__ == "__main__":
    sys.exit(main())
