# Debug Report

Legacy work-item identifiers in this historical report are retained only to cross-reference frozen evidence, artifacts, and checks.

## Symptom

The earlier S10-A integration rerun was blocked by an uninitialized pinned
submodule and then by the absent artifact in the isolated validation worktree.
For the authorized rerun, the exact identity-matched S03-B artifact was read
only from the original QAQ worktree through `QAQ_S03_ARTIFACT`; its
`pytorch_model.bin` SHA-256 was verified, and the artifact-backed S10-A checks
passed.

## Reproduction Command

Working directory: project root
Shell: `bash`
Runtime: `~/.venv/bin/python`, Python 3.12.3
Environment: project virtual environment activated from `~/.venv`
Relevant environment variables:
```text
PYTHONPATH=src:third_party/any-precision-llm:.
QAQ_S03_ARTIFACT=<identity-matched S03-B artifact>
QAQ_MODEL_DEVICE=cuda:0
```

The authorized rerun used the following focused command after checking the
artifact hash against the manifest:

```bash
source ~/.venv/bin/activate
which python
python --version
PYTHONPATH=src:third_party/any-precision-llm:. \
  QAQ_S03_ARTIFACT=<identity-matched S03-B artifact> \
  QAQ_MODEL_DEVICE=cuda:0 pytest -q tests/integration/test_static_six_bit_execution.py
3 passed in 221.42s
```

The static 4/8-bit preservation command passed `10 passed in 640.26s`.

## Expected Behavior

The S10-A integration should load the identity-matched packed S03-B artifact,
exercise pinned CUDA precision-6 execution, and run the full-model static-6
smoke while preserving the existing static paths.

## Actual Behavior

The pinned dependency imported successfully at the exact requested commit.
The identity-matched artifact was available through `QAQ_S03_ARTIFACT`, and
all three S10-A integration tests executed successfully:

```text
3 passed in 221.42s
```

The full-model static-6 smoke was finite and deterministic, and direct imports
of `any_precision` and `any_precision_ext` resolved successfully.

## Error Log

```text
3 passed in 221.42s
10 passed in 640.26s
```

## Failure Layer Classification

Most likely layer: resolved validation-worktree artifact setup.

* Command problem: no
* Permission problem: no
* Shell/script invocation problem: no
* Environment problem: no
* Dependency problem: repaired
* Python/package/import problem: no
* GPU/CUDA problem: no
* Distributed/torchrun problem: no
* Filesystem/path problem: repaired
* Data/checkpoint/model file problem: repaired
* Code logic problem: no evidence
* Configuration problem: no
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: S10-A artifact-backed validation passed after the
identity-matched artifact was supplied read-only.

## Hypotheses

### Hypothesis 1: The Any-Precision submodule is uninitialized

Why it could explain the original symptom: the parent repository had the
gitlink but no checked-out source, so backend validation could not import the
pinned implementation.
Evidence for: the initial submodule status was `-a3257d02740cc5757c78673da534b0630ff3a4ea`; after initialization, the checkout is exactly that commit and clean, and both backend modules import.
Evidence against: none for the original setup failure.
How to verify: completed with `git submodule update --init --recursive third_party/any-precision-llm` and the exact commit/status checks.

### Hypothesis 2: The nested packed-model artifact was unavailable in the isolated worktree (legacy work item S03-B)

Why it explained the earlier symptom: the integration fixtures skip before
checkpoint loading whenever the manifest's local artifact path is absent.
Evidence for: the initial isolated rerun reported the fixture's
artifact-unavailable skip.
Evidence against: the exact identity-matched artifact was subsequently
authorized read-only through `QAQ_S03_ARTIFACT`, its SHA-256 matched the
manifest, and all S10-A checks passed.
How to verify: completed with the focused S10-A command and the static 4/8-bit
preservation result recorded above.

## Most Likely Root Cause

The previously reported dependency and artifact-availability failures were
validation-worktree setup issues. The pinned submodule is clean at commit
`a3257d02740cc5757c78673da534b0630ff3a4ea`, the identity-matched artifact was
verified read-only, and no S10-A implementation assertion failed.

## Minimal Fix

No source-code change was indicated. The exact pinned submodule was initialized
without changing its source or the parent gitlink. The existing identity-matched
artifact was supplied read-only through `QAQ_S03_ARTIFACT`; it was not
regenerated, requantized, or substituted.

## Verification

```bash
source ~/.venv/bin/activate
which python
python --version
git -C third_party/any-precision-llm rev-parse HEAD
git -C third_party/any-precision-llm status --short
git submodule status -- third_party/any-precision-llm
PYTHONPATH=src:. pytest -q tests/unit/test_static_precision_validation.py
PYTHONPATH=src:third_party/any-precision-llm:. \
  QAQ_S03_ARTIFACT=<identity-matched S03-B artifact> \
  QAQ_MODEL_DEVICE=cuda:0 pytest -q tests/integration/test_static_six_bit_execution.py
PYTHONPATH=src:. QAQ_S03_ARTIFACT=<identity-matched S03-B artifact> \
  QAQ_MODEL_DEVICE=cuda:0 pytest -q \
  tests/integration/test_static4_forward.py \
  tests/integration/test_static8_forward.py \
  tests/integration/test_expected_modules_quantized.py \
  tests/integration/test_no_duplicate_precision_models.py \
  tests/integration/test_manifest_byte_count.py \
  tests/integration/test_checkpoint_roundtrip.py
```

Expected verification result: the pinned backend is clean at the recorded
commit, the verified identity-matched artifact is available read-only, and the
artifact-backed S10-A and static 4/8-bit preservation tests pass.

Actual verification result: the submodule is clean at
`a3257d02740cc5757c78673da534b0630ff3a4ea`; the unit test passed `11 passed in
1.71s`; the identity-matched artifact hash matched
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`; the
S10-A integration passed `3 passed in 221.42s`; and the static 4/8-bit
preservation suite passed `10 passed in 640.26s`. Direct imports of
`any_precision` and `any_precision_ext` also passed.

## Broader-validation runner follow-up (legacy work item S10-H1)

### Symptom

The focused S10-H1 test module failed because pre-execution validation could
not find the ignored packed artifact in this fresh worktree.

### Reproduction

```bash
source ~/.venv/bin/activate
which python
python --version
PYTHONPATH=src:. pytest -q tests/unit/test_broader_router_validation.py
```

The first run failed 5 tests with:

```text
PAUSE: identity-matched packed artifact is unavailable: .../quantized/.../pytorch_model.bin
```

After artifact provisioning, validation reached the empty pinned checkout and
failed closed with:

```text
PAUSE: pinned Any-Precision checkout is unavailable
```

### Root Cause and Fix

The fresh worktree omitted both the ignored S03-B artifact and the pinned
Any-Precision submodule contents. The validator correctly failed closed. The
new `scripts/provision_packed_model_artifact.py` hashes the source checkpoint before
creating a worktree-local link, verifies the backend revision, and is covered
by a regression test that rejects unverified bytes.

### Verification

```text
31 passed in 64.56s
```
