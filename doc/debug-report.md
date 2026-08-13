# Debug Report

## Symptom

The previously blocked S10-A integration could not reach the pinned
Any-Precision backend because this validation worktree had an uninitialized
submodule. After restoring that dependency, the focused S10-A test remains
unable to execute because the identity-matched S03-B artifact is absent from
the worktree.

## Reproduction Command

Working directory: project root
Shell: `bash`
Runtime: `~/.venv/bin/python`, Python 3.12.3
Environment: project virtual environment activated from `~/.venv`
Relevant environment variables:
```text
PYTHONPATH=src:third_party/any-precision-llm
QAQ_MODEL_DEVICE=cuda:1
```

The prior round reported the following focused command failing before backend
import because the pinned submodule was uninitialized:

```bash
source ~/.venv/bin/activate
which python
python --version
PYTHONPATH=src:third_party/any-precision-llm:. pytest -q tests/integration/test_s10a_static6.py
```

After the exact submodule initialization requested for this round, the same
focused test was rerun with `-rs`.

## Expected Behavior

The S10-A integration should load the identity-matched packed S03-B artifact,
exercise pinned CUDA precision-6 execution, and run the full-model static-6
smoke while preserving the existing static paths.

## Actual Behavior

The pinned dependency now imports successfully at the exact requested commit,
but all three S10-A integration tests skip because the documented artifact
directory is not present in this worktree:

```text
3 skipped in 1.54s
S03-B artifact is unavailable:
quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64
```

The focused public-precision unit test passes, and direct imports of
`any_precision` and `any_precision_ext` resolve successfully.

## Error Log

```text
SKIPPED [1] tests/integration/test_s10a_static6.py:15: S03-B artifact is unavailable
SKIPPED [1] tests/integration/test_s10a_static6.py:41: S03-B artifact is unavailable
SKIPPED [1] tests/integration/test_s10a_static6.py:87: S03-B artifact is unavailable
3 skipped in 1.54s
```

## Failure Layer Classification

Most likely layer: missing test data/artifact in the validation worktree.

* Command problem: no
* Permission problem: no
* Shell/script invocation problem: no
* Environment problem: no
* Dependency problem: repaired
* Python/package/import problem: no
* GPU/CUDA problem: no
* Distributed/torchrun problem: no
* Filesystem/path problem: yes
* Data/checkpoint/model file problem: yes
* Code logic problem: no evidence
* Configuration problem: no
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: missing identity-matched S03-B artifact; the original
missing-submodule setup failure is repaired.

## Hypotheses

### Hypothesis 1: The Any-Precision submodule is uninitialized

Why it could explain the original symptom: the parent repository had the
gitlink but no checked-out source, so backend validation could not import the
pinned implementation.
Evidence for: the initial submodule status was `-a3257d02740cc5757c78673da534b0630ff3a4ea`; after initialization, the checkout is exactly that commit and clean, and both backend modules import.
Evidence against: none for the original setup failure.
How to verify: completed with `git submodule update --init --recursive third_party/any-precision-llm` and the exact commit/status checks.

### Hypothesis 2: The S03-B artifact is unavailable in this worktree

Why it explains the current symptom: the integration fixtures skip before
checkpoint loading whenever the manifest's local artifact path is absent.
Evidence for: `quantized/` is absent in this worktree and all three S10-A tests report the fixture's artifact-unavailable skip.
Evidence against: the artifact is documented by the repository, but no copy is available within this validation worktree.
How to verify: provide the exact existing artifact within the authorized
validation environment without regenerating or requantizing it, then rerun the
same focused command.

## Most Likely Root Cause

The previously reported dependency failure was a setup issue caused by the
uninitialized pinned submodule and is fixed by checking out commit
`a3257d02740cc5757c78673da534b0630ff3a4ea`. The remaining S10-A evidence is
blocked by the separate absence of the identity-matched S03-B artifact in this
worktree, not by a failing implementation assertion.

## Minimal Fix

No source-code change is indicated. The exact pinned submodule was initialized
without changing its source or the parent gitlink. The remaining focused
integration requires the already-existing artifact; it must not be regenerated,
requantized, or substituted from another checkout.

## Verification

```bash
source ~/.venv/bin/activate
which python
python --version
git -C third_party/any-precision-llm rev-parse HEAD
git -C third_party/any-precision-llm status --short
git submodule status -- third_party/any-precision-llm
PYTHONPATH=src:. pytest -q tests/unit/test_static_precision_validation.py
PYTHONPATH=src:third_party/any-precision-llm:. QAQ_MODEL_DEVICE=cuda:1 pytest -q -rs tests/integration/test_s10a_static6.py
```

Expected verification result: the pinned backend is clean at the recorded
commit, the public-precision unit test passes, and the artifact-backed S10-A
tests execute rather than skip.

Actual verification result: the submodule is clean at
`a3257d02740cc5757c78673da534b0630ff3a4ea`; the unit test passed `11 passed in
1.71s`; the S10-A integration produced `3 skipped in 1.54s` because the
artifact is missing. Direct imports of `any_precision` and
`any_precision_ext` also passed.
