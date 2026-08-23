"""Architecture checks for semantic reusable modules and one-way script adapters."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
SOURCE_ROOT = ROOT / "src" / "qaq"
SCRIPTS_ROOT = ROOT / "scripts"
STAGE_MODULE_NAME = re.compile(r"^s\d+[a-z]?(?:_|$)", re.IGNORECASE)


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return tuple(imported)


def _forbidden_imports(path: Path) -> tuple[str, ...]:
    return tuple(
        name
        for name in _imports(path)
        if name == "scripts"
        or name.startswith(("scripts.", "run_s", "validate_s"))
    )


def test_reusable_modules_and_symbols_have_semantic_names() -> None:
    stage_named_modules = [
        str(path.relative_to(SOURCE_ROOT))
        for path in _python_files(SOURCE_ROOT)
        if STAGE_MODULE_NAME.match(path.stem)
    ]
    stage_named_symbols: list[str] = []
    for path in _python_files(SOURCE_ROOT):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
            elif isinstance(node, ast.Assign):
                names.extend(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.append(node.target.id)
            stage_named_symbols.extend(
                f"{path.relative_to(SOURCE_ROOT)}:{name}"
                for name in names
                if STAGE_MODULE_NAME.match(name)
            )
    assert stage_named_modules == []
    assert stage_named_symbols == []


def test_source_never_imports_scripts() -> None:
    violations = {
        str(path.relative_to(ROOT)): _forbidden_imports(path)
        for path in _python_files(SOURCE_ROOT)
        if _forbidden_imports(path)
    }
    assert violations == {}


def test_scripts_depend_only_on_reusable_source_not_other_scripts() -> None:
    violations = {
        str(path.relative_to(ROOT)): _forbidden_imports(path)
        for path in _python_files(SCRIPTS_ROOT)
        if _forbidden_imports(path)
    }
    assert violations == {}


def test_reusable_router_candidates_use_semantic_symbol() -> None:
    source = "\n".join(path.read_text() for path in _python_files(SOURCE_ROOT))
    assert "S10_CANDIDATE_BITS" not in source
    assert "THREE_WAY_CANDIDATE_BITS" in source
