# replaygate

[![ci](https://github.com/Dowgg18/agent-replay-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/Dowgg18/agent-replay-gate/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-2d6a4f)](LICENSE)

**A pre-deploy regression gate for conversational agents.**

Replay scripted conversations through your real code with every I/O boundary
sealed, fail the build the moment a deterministic path reaches for the model, and
grade whatever is left on a published rubric. One exit code, two artifacts, no
flaky substring assertions.

```bash
$ python -m demo.run_gate
# Replay gate: PASS
- cases: 9
- mean score: 100.0
$ echo $?
0
```

---

## Why agent tests rot

Three failure modes, all of them familiar:

**Substring assertions break on rewording.** `assert "available" in reply` fails
the day someone improves the copy. After the third false alarm the team stops
reading the results, and by then the suite is decoration.

**Live-model tests are non-deterministic and cost money.** They also hide the
thing you most want to know: whether the *deterministic* path is still
deterministic. A rule-based branch that quietly starts calling the LLM gets
slower, costlier and unreproducible, and every assertion still passes.

**"It looked fine in the console" is not a gate.** It does not run on a pull
request and it does not return an exit code.

## What this is, and is not

It is an opinionated gate: cases in, one exit code out, two artifacts on disk.

It is **not** another eval framework. If you want to compare prompts across
models, use promptfoo or langsmith. This answers a narrower question — *did this
commit break the agent?* — and answers it in CI.

---

## The sealed sandbox

Patching every boundary and hoping you got them all fails silently in the
direction that costs most. So a seam is not "patched or not" — it carries a
policy:

```python
with sealed(
    seam("app.deps.store",       Fake(FakeStore(rows=...))),
    seam("app.deps.llm.complete", Forbid("deterministic path must not call the model")),
    seam("app.deps.clock.now",    Count(frozen_clock, label="clock")),
) as sandbox:
    ...
sandbox.crossings   # {"clock": 3}
sandbox.violations  # []
```

| policy | behaviour | use it for |
|---|---|---|
| `Fake` | substitute a double | the ordinary case |
| `Forbid` | raise on call | boundaries this path must never cross |
| `Count` | let it through, record the crossing | allowed but expensive |

`Forbid` is the one that changes code review. A deterministic step that reaches
for an LLM does not produce a slightly worse score — it produces a red run with
the stack trace of the exact call.

`Count` feeds the rubric: falling back to the model is legitimate, and it should
still cost points, because a rubric that scores it as free will watch the habit
spread.

Patches are restored in reverse order in a `finally`. A harness that leaks a
patch fails somewhere unrelated, next week, in a test nobody changed.

## Fakes with declared fidelity

A fake counter that returns `1` forever means the escalation branch never fires
in replay — and that branch is the one most likely to be broken, because it is
the one nobody exercises by hand. `FakeCounter` implements real increment and
expiry. `FakeStore` implements filtering and ordering, so a case asserting *"do
not offer a room that cannot fit"* fails for the right reason, and records writes
so a read-only replay can prove it stayed read-only.

Every fake states what it is faithful to. The honest answer is never
"everything".

## Expectations, not assertions

```yaml
- id: does-not-offer-a-taken-room
  kind: booking
  seed_state: { node: ask_size, date: "2026-09-01", time: "14:00" }
  turns:
    - user: "5 people"
      expect:
        node: offer
        contains_all: ["Atrium"]
        not_contains_any: ["Studio"]     # already booked at that slot
```

An expectation is a claim about observable structure — the node reached, the
state persisted, the intent resolved — with text matching available but never the
only tool. `seed_state` skips ahead so a multi-turn case does not replay twenty
turns to reach the interesting one.

## Scoring: named defects with weights

```
| signal                | weight | absolute |
|-----------------------|--------|----------|
| hallucinated_entity   |     60 | yes      |
| crashed               |    100 | yes      |
| state_corrupted       |     30 | no       |
| escalation_missed     |     25 | no       |
| expectation_failed    |     20 | no       |
| model_fallback        |     20 | no       |
| classifier_fallback   |      8 | no       |
```

Score is `100 - sum(weights)`, bucketed `ok` / `needs-review` / `critical`. The
rubric ships inside the report, so an `84` can always be expanded into the list
of defects that produced it. A score you cannot audit is a number someone learns
to game.

An unknown signal raises rather than scoring as free — that is how a rubric drifts
out of sync with the code that emits it.

## The gate

Two kinds of threshold, because they answer different questions:

- **Absolute-zero** for defects with no acceptable rate. One invented catalog
  entry fails the run at any volume. *"Only 2% of replies hallucinate"* is not a
  passing grade, it is an incident waiting for the right customer.
- **Rates** for everything else, because a gate that demands 100 gets disabled
  within a month.

Pick both from a baseline: run the suite against the current production revision,
take what it scores, set the floor just under it. A gate calibrated against an
aspiration fails on day one and teaches everyone to reach for `--no-verify`.

An empty run fails. A suite that collected nothing is not a passing suite.

## Where the judged tier earns its cost

The deterministic tier cannot catch what nobody thought to name. This is asserted
rather than claimed — the suite contains both halves:

```python
def test_a_denylist_only_catches_what_someone_thought_to_name():
    results = replay_all(cases, HallucinatingAgent, config)
    assert len(caught) == 1        # only the case that lists "Terrace"
    assert outcome.passed is True  # averaged away

def test_the_judge_catches_the_invention_the_denylist_missed():
    ...
    assert outcome.passed is False
    assert outcome.absolute_hits   # hallucinated_entity
```

So the judge is asked one thing the deterministic tier cannot ask: *is this reply
supported by what the agent actually retrieved?* Not "does it sound right" — a
confident invention sounds right every time. Tool results are the ground truth,
temperature is 0, output is JSON.

It **fails open**, and says so. A grader outage is not an agent regression, so a
judge timeout must not turn the build red. Every fail-open is counted in the
report, because a green run built on ten judge timeouts should look different
from a clean one. And it has a budget: the tier stops when the money is spent
rather than quietly invoicing whoever pushed the branch.

---

## The demo

A room-booking assistant for a coworking space, chosen because it has the three
properties that make agents hard to test: multi-turn state that arrives out of
order, a catalog that answers must be grounded in, and a threshold rule that only
fires when the counter has real semantics.

```bash
pip install -e ".[dev,judge]"

python -m demo.run_gate                        # deterministic tier
python -m demo.run_gate --judged               # + judged tier
python -m demo.run_gate --strict-determinism   # any model fallback fails the build
pytest                                         # the harness's own tests
```

### A regression this gate actually caught

Not a hypothetical. The demo agent was written first and the suite second, and
the first run came back `FAIL`, mean 82.2, with three real bugs:

```
[ 80] happy-path-stepwise
        turn 3: node: expected 'ask_time', reached 'ask_size'
[ 40] all-slots-in-one-message
        turn 3: state.time: expected '15:00', got None
        turn 3: missing text: 'Atrium'
```

Both came from the same class of bug in slot parsing. `"2026-09-02"` matched the
time pattern as well as the date one — `09` is a perfectly good hour — so the
agent silently filled `time` from the month and skipped a question. And a message
carrying all three slots only ever filled two, because the headcount branch
returned before the clock was read.

The fix was to consume slots in order of specificity and remove each match from
the text before the next, looser pattern runs. The third finding was a harness
bug, not an agent one: escalation cases were being charged `model_fallback` for
doing exactly what they exist to test, which is why `count_exempt_kinds` exists.

## Limits, honestly

- **The judge fails open by design.** That is the right call for CI and it means
  a sustained grader outage produces a green run. The count is in the report;
  alert on it if you care.
- **A rubric is an opinion.** The weights here are defaults, not findings.
  Calibrate against your own baseline.
- **Judges drift across model versions.** Pin the model, and re-baseline when you
  move it.
- **No cross-turn user simulator.** Cases are scripted; the harness does not
  invent a conversational partner that adapts to the agent's replies.
- **Single process.** Cases run sequentially. Fine at hundreds, not at hundreds
  of thousands.

MIT licensed.
