# Debug Report

## Symptom

The real packed S06 soft-routing regression cannot construct its pinned Any-Precision backend because the submodule directory is present but uninitialized.

## Reproduction Command

Working directory: project root
Shell: `bash`
Runtime: `~/.venv/bin/python`, Python 3.12.3
Environment: project virtual environment activated from `~/.venv`
Relevant environment variables:
```text
PYTHONPATH=src:third_party/any-precision-llm
```

```bash
source ~/.venv/bin/activate
which python
python --version
PYTHONPATH=src:third_party/any-precision-llm pytest -q tests/integration/test_s06_soft_packed.py
```

## Expected Behavior

The focused S06 real packed-backend tests should execute the pinned 4-bit and 8-bit paths and verify soft endpoint parity and gradient propagation.

## Actual Behavior

Two tests failed before backend execution and one artifact-dependent Qwen3 test skipped because the disposable worktree has no S03-B artifact.

## Error Log

```text
RuntimeError: Pinned Any-Precision source is not initialized: third_party/any-precision-llm

2 failed, 1 skipped in 2.86s
```

## Failure Layer Classification

Most likely layer: dependency checkout/setup.

* Command problem: no
* Permission problem: no
* Shell/script invocation problem: no
* Environment problem: yes
* Dependency problem: yes
* Python/package/import problem: no
* GPU/CUDA problem: no
* Distributed/torchrun problem: no
* Filesystem/path problem: yes
* Data/checkpoint/model file problem: no
* Code logic problem: no
* Configuration problem: no
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: missing pinned dependency checkout in the worktree.

## Hypotheses

### Hypothesis 1: The Any-Precision submodule is uninitialized

Why it could explain the symptom: `src/qaq/s01_backend.py` validates that the dependency path is its own Git repository before importing the backend.
Evidence for: `third_party/any-precision-llm` exists but is empty; the reproduction raised the guard's exact “source is not initialized” error.
Evidence against: none observed.
How to verify: restore the recorded submodule revision and rerun the same focused command.

### Hypothesis 2: The pinned backend or CUDA runtime is incompatible

Why it could explain the symptom: the tests require the compiled Any-Precision CUDA backend.
Evidence for: none; execution stopped before importing the backend or checking CUDA behavior.
Evidence against: the failure occurs in the repository-root validation before backend import.
How to verify: rerun after the exact dependency checkout is available.

## Most Likely Root Cause

The worktree contains the parent repository's Any-Precision gitlink but not its checked-out contents, so the backend guard correctly refuses to run the real packed regression.

## Minimal Fix

Populate `third_party/any-precision-llm` at the already recorded commit `a3257d02740cc5757c78673da534b0630ff3a4ea`, without modifying upstream source or the parent gitlink.

## Verification

```bash
source ~/.venv/bin/activate
which python
python --version
PYTHONPATH=src:third_party/any-precision-llm pytest -q tests/integration/test_s06_soft_packed.py
```

Expected verification result: the two synthetic real-packed tests pass; the artifact-dependent Qwen3 test may remain skipped when the S03-B artifact is absent.

Actual verification result: `2 passed, 1 skipped in 12.48s`. The skip was the expected missing S03-B artifact; both real packed tests executed against commit `a3257d02740cc5757c78673da534b0630ff3a4ea` and passed.
