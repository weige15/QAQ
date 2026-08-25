---
description: Evaluate one QAQ stage against its acceptance conditions
argument-hint: "<stage>"
---

Evaluate exactly QAQ stage `$1`.

This is an evidence gate, not an implementation task.

Follow `AGENTS.md`, including its environment requirements and material-decision
boundary.

Read:

- `docs/STATUS.md`;
- `docs/DECISIONS.md`;
- the document for stage `$1` under `docs/stages/`;
- the source notes and papers required by that stage;
- the exact changed code and configuration;
- the exact required tests and commands; and
- the exact result or evidence artifacts required by the stage.

Check every required:

- task;
- invariant;
- acceptance condition;
- prohibited action;
- test;
- output;
- reproducibility requirement;
- unresolved question; and
- stage-specific continue, pause, revise, or stop condition.

Distinguish:

- what current evidence establishes;
- what remains unknown; and
- what is an implementation assumption rather than a source-supported fact.

Where applicable, explicitly verify the QAQ baseline invariants in
`AGENTS.md`, including physical bit-packing, frozen quantized weights during
router training, and transfer of packed planes rather than unpacked weights.

Classify the gate as exactly one of:

### PASS

Use `PASS` only when current authoritative evidence establishes every required
condition for stage `$1` and no required stage work remains.

### PAUSE

Use `PAUSE` when the stage remains valid but cannot currently proceed or be
closed because required external material, resources, credentials, execution,
or other evidence is unavailable.

State exactly what is missing and what evidence would allow work to continue.

### REVISE

Use `REVISE` when current evidence shows that stage `$1` needs a bounded
correction that remains inside its existing scientific meaning, frozen
protocol, scope, and acceptance criteria.

State the failed condition and the smallest correction needed.

### STOP

Use `STOP` when proceeding would require changing a scientific claim, frozen
protocol, threshold, scope, non-goal, preserved behavior, or other material
boundary, or would otherwise require an unauthorized destructive or
irreversible action.

State the material decision that prevents further current-stage work.

Do not implement corrections.

Do not begin a later stage.

Do not modify PDFs under `papers/`.

Update `docs/STATUS.md` before finishing only if the verified gate result
changes durable project state.

Update `docs/DECISIONS.md` only if the gate itself establishes a durable
material decision that belongs there.

Finish with the classification and the evidence supporting it.
