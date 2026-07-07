#!/usr/bin/env python3
"""project_context.py — build a Markdown "map" of this repository for fast onboarding.

The goal is to give a new working session (human or AI) a complete, accurate
picture of the project *without* having to read every source file:

  * what modules exist and what each one is for,
  * the public API of every module (classes, methods, functions — with real
    signatures parsed from the AST, plus first-line docstrings),
  * how the modules depend on each other (internal import graph),
  * where the entrypoints are and how to run them,
  * what the config files and data assets contain.

It parses Python with the standard-library ``ast`` module (no regex guessing,
no third-party deps required), so signatures and structure are exact.

Run it from anywhere:

    python project_context.py                 # writes ./project_context/*.md
    python project_context.py --root . --output project_context
    python project_context.py --print-tree    # also echo the file tree to stdout

Generated files (in the ``project_context/`` directory):

    00_overview.md        Top-level map: stats, file tree, entrypoints, how to run.
    architecture.md       Per-module summary + one-line API (the "where do I look" map).
    api_reference.md      Full symbol listing for every module (the "give me details" map).
    dependencies.md       Internal import graph — imports / imported-by per module.
    configs.md            Config files (YAML/TOML/requirements) and their key fields.
    tests.md              Test files, their test functions, and modules under test.
    data_and_assets.md    Data files, docs and other non-code assets.
    index.json            Machine-readable dump of everything above.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_OUTPUT_DIRNAME = "project_context"

# Directories never worth indexing (dependencies, caches, build/run artifacts).
EXCLUDE_DIRS = {
    ".git", ".idea", ".vscode", ".claude", ".claude_context",
    ".venv", "venv", "env", ".env",
    "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "outputs", "logs", "checkpoints", "runs", "wandb",
    "tmp", "temp", "build", "dist", ".ipynb_checkpoints",
    DEFAULT_OUTPUT_DIRNAME,
}

PY_EXT = {".py"}
CONFIG_EXT = {".yaml", ".yml", ".toml", ".ini", ".cfg"}
DOC_EXT = {".md", ".rst", ".txt"}
DATA_EXT = {".csv", ".json", ".parquet", ".xlsx", ".xls", ".tsv"}
NOTEBOOK_EXT = {".ipynb"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".svg", ".pdf"}
WEIGHT_EXT = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".bin", ".h5"}

MAX_AST_BYTES = 3_000_000     # skip AST parsing of pathologically large files
MAX_VALUE_LEN = 70            # truncate constant / default values in output
MAX_DOC_LEN = 140            # truncate docstring first lines


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class FunctionInfo:
    name: str
    signature: str
    returns: str | None
    decorators: list[str]
    doc: str | None
    lineno: int
    is_async: bool = False


@dataclass
class ClassInfo:
    name: str
    bases: list[str]
    decorators: list[str]
    doc: str | None
    lineno: int
    methods: list[FunctionInfo] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)


@dataclass
class ModuleInfo:
    path: str                       # repo-relative, posix style
    module: str                     # dotted module name
    package: str                    # top-level dir / group (e.g. "src", "scripts", "<root>")
    doc: str | None
    lines: int
    size_bytes: int
    imports: list[str] = field(default_factory=list)
    local_imports: list[str] = field(default_factory=list)   # resolved to repo modules
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    has_main: bool = False
    cli_args: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AssetInfo:
    path: str
    kind: str          # config | doc | data | notebook | image | weights | other
    size_bytes: int
    summary: str = ""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def trim(value: str, max_len: int = MAX_VALUE_LEN) -> str:
    value = " ".join(str(value).replace("\t", " ").split())
    return value if len(value) <= max_len else value[: max_len - 1] + "…"


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def human_size(num: int) -> str:
    step = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if step < 1024 or unit == "GB":
            return f"{step:.0f}{unit}" if unit == "B" else f"{step:.1f}{unit}"
        step /= 1024
    return f"{num}B"


def unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def first_doc_line(node: ast.AST) -> str | None:
    try:
        doc = ast.get_docstring(node, clean=True)
    except Exception:
        doc = None
    if not doc:
        return None
    line = doc.strip().splitlines()[0].strip()
    return trim(line, MAX_DOC_LEN) or None


def module_name_for(rel_posix: str) -> str:
    stem = rel_posix[:-3] if rel_posix.endswith(".py") else rel_posix
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def package_for(rel_posix: str) -> str:
    parts = rel_posix.split("/")
    return parts[0] if len(parts) > 1 else "<root>"


# --------------------------------------------------------------------------- #
# Signature reconstruction (from ast.arguments)
# --------------------------------------------------------------------------- #

def _arg_str(arg: ast.arg, default: ast.AST | None = None) -> str:
    text = arg.arg
    if arg.annotation is not None:
        text += ": " + (unparse(arg.annotation) or "?")
    if default is not None:
        rendered = trim(unparse(default) or "...", MAX_VALUE_LEN)
        text += (" = " if arg.annotation is not None else "=") + rendered
    return text


def render_arguments(args: ast.arguments) -> str:
    parts: list[str] = []
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    n_defaults = len(args.defaults)
    first_default = len(positional) - n_defaults
    posonly_count = len(getattr(args, "posonlyargs", []))

    for i, arg in enumerate(positional):
        default = args.defaults[i - first_default] if i >= first_default else None
        parts.append(_arg_str(arg, default))
        if posonly_count and i == posonly_count - 1:
            parts.append("/")

    if args.vararg is not None:
        parts.append("*" + _arg_str(args.vararg))
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(_arg_str(arg, default))

    if args.kwarg is not None:
        parts.append("**" + _arg_str(args.kwarg))

    return ", ".join(parts)


def build_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo:
    return FunctionInfo(
        name=node.name,
        signature=render_arguments(node.args),
        returns=unparse(node.returns),
        decorators=[unparse(d) or "?" for d in node.decorator_list],
        doc=first_doc_line(node),
        lineno=node.lineno,
        is_async=isinstance(node, ast.AsyncFunctionDef),
    )


# --------------------------------------------------------------------------- #
# Python analysis
# --------------------------------------------------------------------------- #

def _class_attributes(node: ast.ClassDef) -> list[str]:
    attrs: list[str] = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            ann = unparse(item.annotation) or "?"
            if item.value is not None:
                attrs.append(f"{item.target.id}: {ann} = {trim(unparse(item.value) or '...')}")
            else:
                attrs.append(f"{item.target.id}: {ann}")
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    attrs.append(f"{target.id} = {trim(unparse(item.value) or '...')}")
    return attrs


def _module_constants(tree: ast.Module) -> list[str]:
    consts: list[str] = []
    for item in tree.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    consts.append(f"{target.id} = {trim(unparse(item.value) or '...')}")
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if item.target.id.isupper():
                ann = unparse(item.annotation) or "?"
                val = f" = {trim(unparse(item.value) or '...')}" if item.value is not None else ""
                consts.append(f"{item.target.id}: {ann}{val}")
    return consts


def _collect_imports(tree: ast.Module, current_module: str) -> list[str]:
    """Return the dotted names of everything imported by this module."""
    names: list[str] = []
    module_parts = current_module.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module_parts[: max(0, len(module_parts) - node.level)]
            else:
                base = []
            prefix = base + (node.module.split(".") if node.module else [])
            base_dotted = ".".join(prefix)
            for alias in node.names:
                if alias.name == "*":
                    names.append(base_dotted or ".")
                else:
                    names.append(f"{base_dotted}.{alias.name}" if base_dotted else alias.name)
    # de-dup, keep order
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _cli_arguments(tree: ast.Module) -> list[str]:
    flags: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("-")
        ):
            flags.append(node.args[0].value)
    # keep unique, order preserved
    return list(dict.fromkeys(flags))


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                return True
    return False


def analyze_python(path: Path, rel_posix: str) -> ModuleInfo:
    text = read_text(path)
    module_name = module_name_for(rel_posix)
    info = ModuleInfo(
        path=rel_posix,
        module=module_name,
        package=package_for(rel_posix),
        doc=None,
        lines=text.count("\n") + 1 if text else 0,
        size_bytes=path.stat().st_size,
    )

    if path.stat().st_size > MAX_AST_BYTES:
        info.warnings.append("skipped AST parse: file too large")
        return info

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        info.warnings.append(f"syntax error at line {exc.lineno}: {exc.msg}")
        return info

    info.doc = first_doc_line(tree)
    info.imports = _collect_imports(tree, module_name)
    info.constants = _module_constants(tree)
    info.has_main = _has_main_guard(tree)
    info.cli_args = _cli_arguments(tree)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                build_function(item)
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            info.classes.append(
                ClassInfo(
                    name=node.name,
                    bases=[unparse(b) or "?" for b in node.bases],
                    decorators=[unparse(d) or "?" for d in node.decorator_list],
                    doc=first_doc_line(node),
                    lineno=node.lineno,
                    methods=methods,
                    attributes=_class_attributes(node),
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.functions.append(build_function(node))

    if info.lines >= 800:
        info.warnings.append(f"large file: {info.lines} lines")

    return info


def resolve_local_imports(modules: list[ModuleInfo]) -> None:
    """Fill ``local_imports`` on each module with repo modules it depends on."""
    known: set[str] = {m.module for m in modules}
    top_level_pkgs = {m.module.split(".")[0] for m in modules}

    for mod in modules:
        resolved: list[str] = []
        for imported in mod.imports:
            parts = imported.split(".")
            if parts[0] not in top_level_pkgs:
                continue
            # Try progressively shorter prefixes: a.b.c -> a.b -> a
            for cut in range(len(parts), 0, -1):
                candidate = ".".join(parts[:cut])
                if candidate in known and candidate != mod.module:
                    resolved.append(candidate)
                    break
        mod.local_imports = sorted(dict.fromkeys(resolved))


# --------------------------------------------------------------------------- #
# Non-Python asset analysis
# --------------------------------------------------------------------------- #

def classify_asset(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in CONFIG_EXT or path.name in {"requirements.txt", ".gitignore", "Dockerfile"}:
        return "config"
    if ext in DOC_EXT:
        return "doc"
    if ext in DATA_EXT:
        return "data"
    if ext in NOTEBOOK_EXT:
        return "notebook"
    if ext in IMAGE_EXT:
        return "image"
    if ext in WEIGHT_EXT:
        return "weights"
    return "other"


def _summarize_csv(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        cols = ", ".join(header[:12]) + ("…" if len(header) > 12 else "")
        return f"{rows} rows × {len(header)} cols — columns: {cols}"
    except Exception:
        return ""


def _summarize_doc(path: Path) -> str:
    for line in read_text(path).splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return trim(stripped, 120)
    return ""


def analyze_asset(path: Path, rel_posix: str) -> AssetInfo:
    kind = classify_asset(path)
    info = AssetInfo(path=rel_posix, kind=kind, size_bytes=path.stat().st_size)
    ext = path.suffix.lower()
    if ext == ".csv":
        info.summary = _summarize_csv(path)
    elif kind == "doc":
        info.summary = _summarize_doc(path)
    elif path.name == "requirements.txt":
        deps = [ln.split("==")[0].split(">")[0].strip()
                for ln in read_text(path).splitlines() if ln.strip() and not ln.startswith("#")]
        info.summary = ", ".join(deps)
    return info


# --------------------------------------------------------------------------- #
# Repository walk
# --------------------------------------------------------------------------- #

def walk_repo(root: Path, output_dir: Path) -> tuple[list[ModuleInfo], list[AssetInfo]]:
    modules: list[ModuleInfo] = []
    assets: list[AssetInfo] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        current = Path(dirpath)
        if output_dir in (current, *current.parents):
            continue
        for filename in sorted(filenames):
            path = current / filename
            try:
                rel_posix = path.relative_to(root).as_posix()
            except ValueError:
                continue
            if path.suffix.lower() in PY_EXT:
                modules.append(analyze_python(path, rel_posix))
            elif filename == "project_context.py":
                continue
            else:
                assets.append(analyze_asset(path, rel_posix))

    modules.sort(key=lambda m: m.path)
    assets.sort(key=lambda a: a.path)
    resolve_local_imports(modules)
    return modules, assets


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def fn_one_liner(fn: FunctionInfo) -> str:
    prefix = "async def" if fn.is_async else "def"
    ret = f" -> {fn.returns}" if fn.returns else ""
    return f"{prefix} {fn.name}({fn.signature}){ret}"


def render_file_tree(paths: list[str]) -> list[str]:
    tree: dict = {}
    for rel in paths:
        node = tree
        parts = rel.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = None

    lines: list[str] = []

    def walk(node: dict, indent: str) -> None:
        entries = sorted(node.items(), key=lambda kv: (kv[1] is None, kv[0].lower()))
        for name, child in entries:
            lines.append(f"{indent}{name}")
            if child is not None:
                walk(child, indent + "  ")

    walk(tree, "")
    return lines


def write_overview(out: Path, root: Path, modules: list[ModuleInfo],
                   assets: list[AssetInfo], generated: str) -> None:
    total_lines = sum(m.lines for m in modules)
    n_classes = sum(len(m.classes) for m in modules)
    n_functions = sum(len(m.functions) for m in modules)
    n_methods = sum(len(c.methods) for m in modules for c in m.classes)
    entrypoints = [m for m in modules if m.has_main]

    doc_intro = ""
    readme = root / "README.md"
    if readme.exists():
        doc_intro = _summarize_doc(readme)

    all_paths = [m.path for m in modules] + [a.path for a in assets]

    lines = [
        f"# Project context — {root.name}",
        "",
        f"_Generated by `project_context.py` on {generated}._",
        "",
        "**Read this file first.** It is a generated map of the repository so a new "
        "session can navigate straight to the right file instead of scanning everything.",
        "",
    ]
    if doc_intro:
        lines += [f"> {doc_intro}", ""]

    lines += [
        "## Context files in this directory",
        "",
        "| File | Use it to find… |",
        "|---|---|",
        "| `00_overview.md` | This map — stats, file tree, entrypoints, how to run. |",
        "| `architecture.md` | What each module does + its one-line API (start here to locate code). |",
        "| `api_reference.md` | Full signatures of every class/method/function (the details). |",
        "| `dependencies.md` | Which modules import which (internal graph, imported-by). |",
        "| `configs.md` | Config YAMLs, requirements, and their key fields. |",
        "| `tests.md` | Test files, test functions, and the modules they exercise. |",
        "| `data_and_assets.md` | Data files, docs, images and other non-code assets. |",
        "| `index.json` | Machine-readable dump of all of the above. |",
        "",
        "## At a glance",
        "",
        f"- Python modules: **{len(modules)}**  ({total_lines} source lines)",
        f"- Classes: **{n_classes}**  ·  Top-level functions: **{n_functions}**  ·  Methods: **{n_methods}**",
        f"- Non-code assets: **{len(assets)}**",
        f"- Entrypoints (`__main__` guard): **{len(entrypoints)}**",
        "",
    ]

    if entrypoints:
        lines += ["## Entrypoints & how to run", ""]
        for mod in entrypoints:
            desc = f" — {mod.doc}" if mod.doc else ""
            lines.append(f"### `{mod.path}`{desc}")
            lines.append("")
            lines.append(f"```bash\npython {mod.path.replace('/', '/')}\n```")
            if mod.cli_args:
                lines.append("")
                lines.append("CLI flags: " + ", ".join(f"`{a}`" for a in mod.cli_args))
            lines.append("")

    lines += ["## File tree", "", "```"]
    lines += render_file_tree(all_paths)
    lines += ["```", ""]

    write_md(out / "00_overview.md", lines)


def write_architecture(out: Path, modules: list[ModuleInfo]) -> None:
    lines = [
        "# Architecture — module map",
        "",
        "One block per Python module: its purpose, plus a one-line view of every "
        "class and top-level function. Use this to decide *which* file to open; "
        "see `api_reference.md` for full signatures.",
        "",
    ]

    by_pkg: dict[str, list[ModuleInfo]] = {}
    for mod in modules:
        by_pkg.setdefault(mod.package, []).append(mod)

    for pkg in sorted(by_pkg):
        lines += [f"## `{pkg}`", ""]
        for mod in by_pkg[pkg]:
            lines.append(f"### `{mod.path}`")
            if mod.doc:
                lines.append(f"_{mod.doc}_")
            lines.append("")
            meta = [f"{mod.lines} lines"]
            if mod.local_imports:
                meta.append("imports: " + ", ".join(f"`{d}`" for d in mod.local_imports))
            lines.append("- " + "  ·  ".join(meta))
            if mod.constants:
                lines.append("- constants: " + ", ".join(f"`{c}`" for c in mod.constants[:12]))

            for cls in mod.classes:
                bases = f"({', '.join(cls.bases)})" if cls.bases else ""
                doc = f" — {cls.doc}" if cls.doc else ""
                lines.append(f"- **class `{cls.name}{bases}`**{doc}")
                for method in cls.methods:
                    if method.name.startswith("_") and method.name != "__init__":
                        continue
                    mdoc = f" — {method.doc}" if method.doc else ""
                    lines.append(f"    - `{method.name}(…)`{mdoc}")

            for fn in mod.functions:
                if fn.name.startswith("_"):
                    continue
                doc = f" — {fn.doc}" if fn.doc else ""
                lines.append(f"- `{fn.name}(…)`{doc}")

            lines.append("")

    write_md(out / "architecture.md", lines)


def write_api_reference(out: Path, modules: list[ModuleInfo]) -> None:
    lines = [
        "# API reference",
        "",
        "Full signatures for every module, class, method and function "
        "(parsed from the AST — these are exact).",
        "",
    ]
    for mod in modules:
        lines.append(f"## `{mod.path}`")
        if mod.doc:
            lines.append(f"_{mod.doc}_")
        lines.append("")
        if mod.warnings:
            lines.append("> ⚠️ " + "; ".join(mod.warnings))
            lines.append("")
        if mod.constants:
            lines.append("**Constants**")
            for const in mod.constants:
                lines.append(f"- `{const}`")
            lines.append("")

        for cls in mod.classes:
            bases = f"({', '.join(cls.bases)})" if cls.bases else ""
            deco = "".join(f"@{d} " for d in cls.decorators)
            lines.append(f"### {deco}class `{cls.name}{bases}`")
            if cls.doc:
                lines.append(f"_{cls.doc}_")
            if cls.attributes:
                lines.append("")
                lines.append("Attributes:")
                for attr in cls.attributes:
                    lines.append(f"- `{attr}`")
            if cls.methods:
                lines.append("")
                lines.append("Methods:")
                for method in cls.methods:
                    lines.append(f"- `{fn_one_liner(method)}`"
                                 + (f" — {method.doc}" if method.doc else ""))
            lines.append("")

        if mod.functions:
            lines.append("### Functions")
            for fn in mod.functions:
                deco = "".join(f"@{d} " for d in fn.decorators)
                lines.append(f"- `{deco}{fn_one_liner(fn)}`"
                             + (f" — {fn.doc}" if fn.doc else ""))
            lines.append("")

    write_md(out / "api_reference.md", lines)


def write_dependencies(out: Path, modules: list[ModuleInfo]) -> None:
    imported_by: dict[str, list[str]] = {}
    for mod in modules:
        for dep in mod.local_imports:
            imported_by.setdefault(dep, []).append(mod.module)

    lines = [
        "# Internal dependency graph",
        "",
        "Only imports that resolve to modules *inside this repo* are shown.",
        "",
        "## Imports (module → what it uses)",
        "",
    ]
    for mod in modules:
        if mod.local_imports:
            lines.append(f"- `{mod.module}` → " + ", ".join(f"`{d}`" for d in mod.local_imports))
    lines += ["", "## Imported-by (module → what depends on it)", ""]
    for module_name in sorted(imported_by):
        users = sorted(set(imported_by[module_name]))
        lines.append(f"- `{module_name}` ← " + ", ".join(f"`{u}`" for u in users))

    write_md(out / "dependencies.md", lines)


def write_tests(out: Path, modules: list[ModuleInfo]) -> None:
    tests = [m for m in modules if m.path.startswith("tests/") or Path(m.path).name.startswith("test_")]
    lines = [
        "# Tests",
        "",
        f"{len(tests)} test module(s). Each lists its test functions and the repo "
        "modules it imports (i.e. what it exercises).",
        "",
    ]
    for mod in tests:
        lines.append(f"## `{mod.path}`")
        if mod.doc:
            lines.append(f"_{mod.doc}_")
        if mod.local_imports:
            lines.append(f"- exercises: " + ", ".join(f"`{d}`" for d in mod.local_imports))
        test_fns = [f for f in mod.functions if f.name.startswith("test")]
        test_methods = [(c.name, m) for c in mod.classes for m in c.methods if m.name.startswith("test")]
        for fn in test_fns:
            lines.append(f"- `{fn.name}`" + (f" — {fn.doc}" if fn.doc else ""))
        for cls_name, method in test_methods:
            lines.append(f"- `{cls_name}.{method.name}`" + (f" — {method.doc}" if method.doc else ""))
        lines.append("")
    write_md(out / "tests.md", lines)


def _yaml_key_outline(path: Path, max_lines: int = 60) -> list[str]:
    """Show top-level YAML structure without requiring PyYAML."""
    outline: list[str] = []
    for raw in read_text(path).splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if indent <= 2 and ":" in raw:
            outline.append(raw.rstrip())
        if len(outline) >= max_lines:
            outline.append("  …")
            break
    return outline


def write_configs(out: Path, root: Path, assets: list[AssetInfo]) -> None:
    configs = [a for a in assets if a.kind == "config"]
    lines = ["# Config & tooling files", ""]
    for asset in configs:
        path = root / asset.path
        lines.append(f"## `{asset.path}`")
        if asset.summary:
            lines.append(asset.summary if asset.path.endswith("requirements.txt")
                         else f"_{asset.summary}_")
        if path.suffix.lower() in {".yaml", ".yml"}:
            outline = _yaml_key_outline(path)
            if outline:
                lines.append("")
                lines.append("```yaml")
                lines += outline
                lines.append("```")
        lines.append("")
    write_md(out / "configs.md", lines)


def write_data_and_assets(out: Path, assets: list[AssetInfo]) -> None:
    groups = ("data", "doc", "notebook", "image", "weights", "other")
    titles = {
        "data": "Data files", "doc": "Documentation", "notebook": "Notebooks",
        "image": "Images / figures", "weights": "Model weights", "other": "Other files",
    }
    lines = ["# Data & non-code assets", ""]
    for kind in groups:
        items = [a for a in assets if a.kind == kind]
        if not items:
            continue
        lines += [f"## {titles[kind]} ({len(items)})", ""]
        for asset in items:
            suffix = f" — {asset.summary}" if asset.summary else ""
            lines.append(f"- `{asset.path}` ({human_size(asset.size_bytes)}){suffix}")
        lines.append("")
    write_md(out / "data_and_assets.md", lines)


def write_index_json(out: Path, modules: list[ModuleInfo],
                     assets: list[AssetInfo], generated: str) -> None:
    payload = {
        "generated": generated,
        "modules": [asdict(m) for m in modules],
        "assets": [asdict(a) for a in assets],
    }
    (out / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def build(root: Path, output_dir: Path, print_tree: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    modules, assets = walk_repo(root, output_dir)

    write_overview(output_dir, root, modules, assets, generated)
    write_architecture(output_dir, modules)
    write_api_reference(output_dir, modules)
    write_dependencies(output_dir, modules)
    write_tests(output_dir, modules)
    write_configs(output_dir, root, assets)
    write_data_and_assets(output_dir, assets)
    write_index_json(output_dir, modules, assets, generated)

    print(f"[project_context] root       : {root}")
    print(f"[project_context] output dir : {output_dir}")
    print(f"[project_context] modules    : {len(modules)}")
    print(f"[project_context] assets     : {len(assets)}")
    print(f"[project_context] start file : {output_dir / '00_overview.md'}")

    if print_tree:
        print("\n".join(render_file_tree([m.path for m in modules] + [a.path for a in assets])))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown context maps for this repository."
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Repository root to scan (default: this script's directory).",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_DIRNAME,
        help=f"Output directory for context files (default: {DEFAULT_OUTPUT_DIRNAME}/).",
    )
    parser.add_argument(
        "--print-tree", action="store_true",
        help="Also print the discovered file tree to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    build(root, output, print_tree=args.print_tree)


if __name__ == "__main__":
    main()