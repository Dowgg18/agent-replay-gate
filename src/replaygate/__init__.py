"""replaygate — a pre-deploy regression gate for conversational agents."""

from .agent import AgentReply, AgentUnderTest
from .cases import Case, Expectation, Turn
from .cases import from_dict as case_from_dict
from .cases import load_yaml as load_cases
from .fakes import FakeCounter, FakeStore, RecordedModel, UnexpectedWrite
from .gate import GateOutcome, Thresholds, evaluate
from .judge import Judge, ScriptedJudge, Verdict
from .report import to_json, to_markdown
from .rubric import DEFAULT_RUBRIC, CaseResult, Rubric, Signal
from .rubric import load_yaml as load_rubric
from .runner import ReplayConfig, replay_all, replay_case
from .sandbox import (
    Count,
    Fake,
    Forbid,
    SandboxReport,
    Seam,
    SeamViolation,
    forbid_model,
    sealed,
    seam,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_RUBRIC",
    "AgentReply",
    "AgentUnderTest",
    "Case",
    "CaseResult",
    "Count",
    "Expectation",
    "Fake",
    "FakeCounter",
    "FakeStore",
    "Forbid",
    "GateOutcome",
    "Judge",
    "RecordedModel",
    "ReplayConfig",
    "Rubric",
    "SandboxReport",
    "ScriptedJudge",
    "Seam",
    "SeamViolation",
    "Signal",
    "Thresholds",
    "Turn",
    "UnexpectedWrite",
    "Verdict",
    "__version__",
    "case_from_dict",
    "evaluate",
    "forbid_model",
    "load_cases",
    "load_rubric",
    "replay_all",
    "replay_case",
    "sealed",
    "seam",
    "to_json",
    "to_markdown",
]
