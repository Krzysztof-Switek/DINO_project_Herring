from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import re
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / ".claude_context"
README = OUTPUT_DIR / "README.md"

INCLUDE_EXT = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".toml",
    ".md",
}

EXCLUDE_DIR_NAMES = {
    ".git", ".idea", ".vscode", ".venv", "venv", "env",
    "__pycache__", "node_modules", "dist", "build", "coverage",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".firebase",
    ".next", ".nuxt", ".svelte-kit", ".turbo", ".cache",
    ".claude_context",
    "checkpoints", "weights", "models_cache", "runs", "outputs",
    "logs", "tmp", "temp",
}

EXCLUDE_FILE_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "firebase-debug.log",
    "firestore-debug.log",
}

MAX_LINE_LEN = 140
MAX_ITEMS = 40
MAX_TEXT_SCAN_CHARS = 8000
LARGE_FILE_LINES = 500
VERY_LARGE_FILE_LINES = 1200


@dataclass
class FileInfo:
    path: str
    suffix: str
    lines: int
    size_bytes: int
    kind: str
    roles: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    local_imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    entrypoint_hints: list[str] = field(default_factory=list)
    data_access_hints: list[str] = field(default_factory=list)
    framework_hints: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def trim(value: str, max_len: int = MAX_LINE_LEN) -> str:
    value = " ".join(str(value).replace("\t", " ").split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def rel_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def unique_sorted(values: Iterable[str], limit: int | None = MAX_ITEMS) -> list[str]:
    result = sorted({trim(v) for v in values if v and trim(v)})
    if limit is None:
        return result
    return result[:limit]


def should_exclude(path: Path) -> bool:
    if path.name in EXCLUDE_FILE_NAMES:
        return True

    if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
        return True

    if path.suffix.lower() not in INCLUDE_EXT:
        return True

    return False


def is_probably_generated(path: Path) -> bool:
    rel = rel_path(path).lower()
    generated_markers = [
        ".min.js",
        ".bundle.js",
        ".generated.",
        ".gen.",
        "generated/",
        "vendor/",
        "third_party/",
    ]
    return any(marker in rel for marker in generated_markers)


def extract_imports(text: str, suffix: str) -> list[str]:
    results: list[str] = []

    patterns = [
        r"^\s*import\s+.+?\s+from\s+['\"](.+?)['\"]",
        r"^\s*import\s+['\"](.+?)['\"]",
        r"^\s*export\s+.+?\s+from\s+['\"](.+?)['\"]",
        r"^\s*const\s+.+?\s*=\s*require\(['\"](.+?)['\"]\)",
        r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+.+",
        r"^\s*import\s+([A-Za-z0-9_\.]+)",
        r"^\s*#include\s+[<\"](.+?)[>\"]",
    ]

    for line in text.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                results.append(match.group(1))

    return unique_sorted(results)


def is_local_import(import_path: str) -> bool:
    return (
        import_path.startswith(".")
        or import_path.startswith("/")
        or import_path.startswith("@/")
        or import_path.startswith("src/")
        or import_path.startswith("app/")
    )


def extract_exports(text: str) -> list[str]:
    patterns = [
        r"export\s+(?:async\s+)?function\s+([A-Za-z0-9_]+)",
        r"export\s+const\s+([A-Za-z0-9_]+)",
        r"export\s+class\s+([A-Za-z0-9_]+)",
        r"exports\.([A-Za-z0-9_]+)\s*=",
        r"module\.exports\.([A-Za-z0-9_]+)\s*=",
        r"__all__\s*=\s*\[([^\]]+)\]",
    ]

    results: list[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if "," in match:
                results.extend(re.findall(r"['\"]([^'\"]+)['\"]", match))
            else:
                results.append(match)

    return unique_sorted(results)


def extract_symbols(text: str) -> list[str]:
    patterns = [
        r"function\s+([A-Za-z0-9_]+)\s*\(",
        r"async\s+function\s+([A-Za-z0-9_]+)\s*\(",
        r"const\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\(",
        r"class\s+([A-Za-z0-9_]+)",
        r"def\s+([A-Za-z0-9_]+)\s*\(",
        r"async\s+def\s+([A-Za-z0-9_]+)\s*\(",
    ]

    results: list[str] = []
    for pattern in patterns:
        results.extend(re.findall(pattern, text))

    return unique_sorted(results)


def extract_routes(text: str) -> list[str]:
    results: list[str] = []

    route_patterns = [
        r"\bapp\.(get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)['\"]",
        r"\brouter\.(get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)['\"]",
        r"@(?:app|router)\.(get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)['\"]",
        r"@api_view\s*\(\s*\[([^\]]+)\]\s*\)",
        r"path\s*\(\s*['\"]([^'\"]+)['\"]",
        r"re_path\s*\(\s*['\"]([^'\"]+)['\"]",
        r"Route::(get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)['\"]",
    ]

    for pattern in route_patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                if len(match) == 2:
                    results.append(f"{match[0].upper()} {match[1]}")
                else:
                    results.append(" ".join(match))
            else:
                results.append(match)

    client_patterns = [
        r"\bfetch\s*\(\s*['\"]([^'\"]+)['\"]",
        r"\baxios\.(get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)['\"]",
        r"\brequests\.(get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)['\"]",
        r"\bhttpx\.(get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)['\"]",
        r"['\"](/api/[^'\"]+)['\"]",
        r"['\"](api/[^'\"]+)['\"]",
    ]

    for pattern in client_patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                results.append(f"{match[0].upper()} {match[1]}")
            else:
                results.append(match)

    return unique_sorted(results)


def detect_frameworks(path: Path, text: str) -> list[str]:
    rel = rel_path(path).lower()
    blob = f"{rel}\n{text[:MAX_TEXT_SCAN_CHARS].lower()}"

    hints: list[str] = []

    checks = {
        "react": ["react", "usestate", "useeffect", ".tsx", ".jsx"],
        "nextjs": ["next.config", "getserversideprops", "app/page.", "pages/api"],
        "vue": ["vue", "definecomponent", "<template>"],
        "svelte": ["svelte", ".svelte"],
        "express": ["express()", "app.get(", "router.get("],
        "fastapi": ["fastapi", "apirouter", "@app.get", "@router.get"],
        "flask": ["flask", "@app.route"],
        "django": ["django", "urlpatterns", "models.model"],
        "firebase": ["firebase", "onrequest", "firebase.json"],
        "pytest": ["pytest", "def test_"],
        "jest/vitest": ["describe(", "it(", "test("],
        "pytorch": ["torch", "nn.module", "dataloader"],
        "tensorflow": ["tensorflow", "keras"],
        "pandas": ["pandas", "dataframe"],
        "sqlalchemy": ["sqlalchemy", "declarative_base", "sessionmaker"],
    }

    for name, markers in checks.items():
        if any(marker in blob for marker in markers):
            hints.append(name)

    return unique_sorted(hints)


def detect_entrypoints(path: Path, text: str) -> list[str]:
    rel = rel_path(path).lower()
    name = path.name.lower()
    hints: list[str] = []

    entry_names = {
        "main.py",
        "app.py",
        "server.py",
        "index.js",
        "index.ts",
        "main.js",
        "main.ts",
        "manage.py",
        "train.py",
        "predict.py",
        "cli.py",
    }

    if name in entry_names:
        hints.append("entrypoint filename")

    if "if __name__ == \"__main__\"" in text or "if __name__ == '__main__'" in text:
        hints.append("python main guard")

    if "app.listen(" in text:
        hints.append("node server listen")

    if "uvicorn.run(" in text:
        hints.append("uvicorn run")

    if "create_app(" in text:
        hints.append("app factory")

    if rel.endswith("package.json"):
        hints.append("package manifest")

    if rel.endswith("pyproject.toml") or rel.endswith("requirements.txt"):
        hints.append("python dependency manifest")

    return unique_sorted(hints)


def detect_data_access(text: str) -> list[str]:
    blob = text[:MAX_TEXT_SCAN_CHARS].lower()
    hints: list[str] = []

    checks = {
        "sql": ["select ", "insert into", "update ", "delete from"],
        "orm": ["session.query", ".objects.filter", ".objects.get", "prisma.", "sequelize"],
        "firestore": ["firestore", ".collection(", ".doc("],
        "mongodb": ["mongodb", "mongoose", "pymongo"],
        "redis": ["redis"],
        "file_io": ["open(", "read_text(", "write_text(", "read_csv", "to_csv"],
        "http_client": ["fetch(", "axios.", "requests.", "httpx."],
    }

    for name, markers in checks.items():
        if any(marker in blob for marker in markers):
            hints.append(name)

    return unique_sorted(hints)


def extract_keywords(path: Path, text: str) -> list[str]:
    rel = rel_path(path).lower()
    blob = f"{rel}\n{text[:MAX_TEXT_SCAN_CHARS].lower()}"

    candidates = [
        "auth", "login", "user", "admin", "role", "permission",
        "api", "route", "handler", "service", "controller",
        "database", "db", "model", "schema", "migration",
        "test", "fixture", "mock", "e2e",
        "config", "settings", "env",
        "cache", "queue", "job", "task", "worker",
        "email", "notification", "webhook",
        "payment", "billing", "invoice",
        "upload", "storage", "file",
        "map", "calendar", "reservation", "booking",
        "train", "predict", "dataset", "loss", "metric",
        "frontend", "component", "state", "form",
    ]

    return [kw for kw in candidates if kw in blob]


def classify_file(path: Path, text: str) -> tuple[str, list[str]]:
    rel = rel_path(path).lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    blob = text[:MAX_TEXT_SCAN_CHARS].lower()

    roles: list[str] = []

    def has_any(markers: list[str]) -> bool:
        return any(marker in blob or marker in rel for marker in markers)

    if (
        "test" in rel
        or name.startswith("test_")
        or name.endswith(".test.ts")
        or name.endswith(".spec.ts")
        or "def test_" in blob
        or "describe(" in blob
    ):
        roles.append("test")

    if extract_routes(text):
        roles.append("route_or_api")

    if has_any(["handler", "controller", "route", "router"]):
        roles.append("handler_controller")

    if has_any(["service", "usecase", "use_case", "manager"]):
        roles.append("service_logic")

    if has_any(["model", "schema", "entity", "repository", "dao"]):
        roles.append("data_model")

    if has_any(["react", "usestate", "useeffect", "<template>", "component"]):
        roles.append("frontend")

    if suffix in {".html", ".css", ".scss"}:
        roles.append("frontend_asset")

    if has_any(["train", "predict", "dataset", "dataloader", "loss", "metric", "torch", "keras"]):
        roles.append("ml_data")

    if name in {
        "package.json", "pyproject.toml", "requirements.txt",
        "dockerfile", "docker-compose.yml", "firebase.json",
        "tsconfig.json", "vite.config.ts", "next.config.js",
    } or "config" in rel or "settings" in rel:
        roles.append("config")

    if rel.startswith("tools/") or rel.startswith("scripts/") or "script" in rel:
        roles.append("tooling")

    if suffix == ".md":
        roles.append("docs")

    if not roles:
        roles.append("other")

    priority = [
        "test",
        "route_or_api",
        "handler_controller",
        "service_logic",
        "data_model",
        "frontend",
        "frontend_asset",
        "ml_data",
        "config",
        "tooling",
        "docs",
        "other",
    ]

    kind = next((role for role in priority if role in roles), "other")
    return kind, unique_sorted(roles, None)


def resolve_local_import(current_file: str, import_path: str, known_paths: set[str]) -> str | None:
    if not is_local_import(import_path):
        return None

    current = ROOT / current_file

    if import_path.startswith("."):
        raw = (current.parent / import_path).resolve()
    else:
        raw = (ROOT / import_path).resolve()

    candidates: list[Path] = []

    for ext in [".py", ".ts", ".tsx", ".js", ".jsx", ".json"]:
        candidates.append(raw.with_suffix(ext))

    for ext in [".py", ".ts", ".tsx", ".js", ".jsx", ".json"]:
        candidates.append(raw / f"index{ext}")
        candidates.append(raw / f"__init__{ext}")

    for candidate in candidates:
        try:
            rel = rel_path(candidate)
        except ValueError:
            continue

        if rel in known_paths:
            return rel

    return None


def analyze_file(path: Path) -> FileInfo:
    text_full = read_text(path)
    text = text_full[:MAX_TEXT_SCAN_CHARS]
    line_count = text_full.count("\n") + 1

    kind, roles = classify_file(path, text)
    imports = extract_imports(text, path.suffix.lower())

    info = FileInfo(
        path=rel_path(path),
        suffix=path.suffix.lower(),
        lines=line_count,
        size_bytes=path.stat().st_size,
        kind=kind,
        roles=roles,
        imports=imports,
        local_imports=[imp for imp in imports if is_local_import(imp)],
        exports=extract_exports(text),
        symbols=extract_symbols(text),
        routes=extract_routes(text),
        entrypoint_hints=detect_entrypoints(path, text),
        data_access_hints=detect_data_access(text),
        framework_hints=detect_frameworks(path, text),
        keywords=extract_keywords(path, text),
    )

    if line_count >= VERY_LARGE_FILE_LINES:
        info.warnings.append(f"very large file: {line_count} lines")
    elif line_count >= LARGE_FILE_LINES:
        info.warnings.append(f"large file: {line_count} lines")

    if is_probably_generated(path):
        info.warnings.append("probably generated/vendor/bundled")

    return info


def build_dependency_edges(files: list[FileInfo]) -> list[dict[str, str]]:
    known_paths = {f.path for f in files}
    edges: list[dict[str, str]] = []

    for file in files:
        for import_path in file.local_imports:
            resolved = resolve_local_import(file.path, import_path, known_paths)
            if resolved:
                edges.append({"source": file.path, "target": resolved})

    return edges


def append_list(lines: list[str], values: list[str], indent: str = "- ") -> None:
    for value in values:
        lines.append(f"{indent}{trim(value)}")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_file_details(lines: list[str], file: FileInfo) -> None:
    lines.append(f"## `{file.path}`")
    lines.append("")
    lines.append(f"- kind: `{file.kind}`")
    lines.append(f"- roles: {', '.join(file.roles)}")
    lines.append(f"- lines: {file.lines}")
    lines.append(f"- size_bytes: {file.size_bytes}")

    sections = [
        ("warnings", file.warnings),
        ("framework hints", file.framework_hints),
        ("entrypoint hints", file.entrypoint_hints),
        ("data access hints", file.data_access_hints),
        ("keywords", file.keywords),
        ("exports", file.exports),
        ("symbols", file.symbols),
        ("routes/api hints", file.routes),
        ("local imports", file.local_imports),
    ]

    for title, values in sections:
        if values:
            lines.append(f"- {title}:")
            append_list(lines, values, "  - ")

    lines.append("")


def write_readme(files: list[FileInfo]) -> None:
    total_lines = sum(f.lines for f in files)
    large_files = [f for f in files if f.lines >= LARGE_FILE_LINES]
    frameworks = unique_sorted(
        fw for file in files for fw in file.framework_hints
    )
    entrypoints = [f for f in files if f.entrypoint_hints]

    lines = [
        "# Claude Code Context",
        "",
        "Read this file first.",
        "",
        "## Mandatory operating rules",
        "",
        "1. Do not read the whole repository.",
        "2. Do not use broad glob searches such as `**/*`.",
        "3. Use these context files as the navigation map.",
        "4. Efficient context use is the goal, not blind minimalism.",
        "5. Reading a few extra relevant functions is acceptable.",
        "6. Reading half of the repo to answer one question is not acceptable.",
        "7. Before opening source files, state:",
        "   - which files you want to open,",
        "   - why each file is needed,",
        "   - what specific information you expect to find.",
        "8. Open at most 3 source files in the first step.",
        "9. Ask for the next specific file only if needed.",
        "10. Modify at most 1 file per implementation step.",
        "11. After each modification, show the diff and one concrete test command.",
        "",
        "## Context files",
        "",
        "- `context_entrypoints.md` — likely app starts, manifests and main files.",
        "- `context_routes.md` — API routes, endpoints and client API calls.",
        "- `context_backend.md` — handlers, services, models and backend logic.",
        "- `context_frontend.md` — UI, components, pages and frontend assets.",
        "- `context_tests.md` — tests and test helpers.",
        "- `context_ml_data.md` — ML, datasets, training and data-processing files.",
        "- `context_config_tooling.md` — config, scripts, manifests and tooling.",
        "- `context_keywords.md` — feature keyword to file mapping.",
        "- `context_dependencies.json` — resolved local import edges.",
        "- `context_files.json` — full machine-readable file index.",
        "",
        "## Recommended workflow",
        "",
        "If task mentions an endpoint or API:",
        "",
        "1. Read `context_routes.md`.",
        "2. Pick likely route/handler/service/test files.",
        "3. Open only 1-3 source files.",
        "",
        "If task mentions a feature but no endpoint:",
        "",
        "1. Read `context_keywords.md`.",
        "2. Then read the matching context file.",
        "3. Open only the most likely source files.",
        "",
        "If task is unclear:",
        "",
        "1. Read `context_entrypoints.md`.",
        "2. Read one targeted context file.",
        "3. Propose a small file-opening plan.",
        "",
        "## Project summary",
        "",
        f"- indexed files: {len(files)}",
        f"- indexed source lines: {total_lines}",
        f"- large files: {len(large_files)}",
        "",
    ]

    if frameworks:
        lines.append("## Detected framework hints")
        lines.append("")
        append_list(lines, frameworks)
        lines.append("")

    if entrypoints:
        lines.append("## Likely entrypoints")
        lines.append("")
        for file in sorted(entrypoints, key=lambda f: f.path):
            lines.append(f"- `{file.path}`")
            append_list(lines, file.entrypoint_hints, "  - ")
        lines.append("")

    if large_files:
        lines.append("## Large files warning")
        lines.append("")
        for file in sorted(large_files, key=lambda f: f.lines, reverse=True):
            lines.append(f"- `{file.path}` — {file.lines} lines")
        lines.append("")

    write_markdown(README, lines)


def write_group_file(filename: str, title: str, files: list[FileInfo], predicate) -> None:
    selected = [file for file in files if predicate(file)]

    lines = [
        f"# {title}",
        "",
        f"Files indexed here: {len(selected)}",
        "",
    ]

    if not selected:
        lines.append("- No files in this group.")
    else:
        for file in sorted(selected, key=lambda f: f.path):
            append_file_details(lines, file)

    write_markdown(OUTPUT_DIR / filename, lines)


def write_routes(files: list[FileInfo]) -> None:
    route_files = [f for f in files if f.routes or "route_or_api" in f.roles]

    lines = [
        "# Routes and API Context",
        "",
        "Use this file first for endpoint, API, handler or client request tasks.",
        "",
        f"Files indexed here: {len(route_files)}",
        "",
    ]

    for file in sorted(route_files, key=lambda f: f.path):
        append_file_details(lines, file)

    write_markdown(OUTPUT_DIR / "context_routes.md", lines)


def write_keywords(files: list[FileInfo]) -> None:
    keyword_map: dict[str, list[str]] = {}

    for file in files:
        for keyword in file.keywords:
            keyword_map.setdefault(keyword, []).append(file.path)

    lines = [
        "# Keyword Index",
        "",
        "Use this file when the task is described by a feature name.",
        "",
    ]

    for keyword in sorted(keyword_map):
        lines.append(f"## {keyword}")
        for path in sorted(set(keyword_map[keyword])):
            lines.append(f"- `{path}`")
        lines.append("")

    write_markdown(OUTPUT_DIR / "context_keywords.md", lines)


def write_json_files(files: list[FileInfo]) -> None:
    dependencies = build_dependency_edges(files)

    (OUTPUT_DIR / "context_files.json").write_text(
        json.dumps([asdict(f) for f in files], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "context_dependencies.json").write_text(
        json.dumps(dependencies, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    files: list[FileInfo] = []

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if should_exclude(path):
            continue

        files.append(analyze_file(path))

    write_readme(files)

    write_group_file(
        "context_entrypoints.md",
        "Entrypoints and Manifests",
        files,
        lambda f: bool(f.entrypoint_hints) or "config" in f.roles,
    )

    write_routes(files)

    write_group_file(
        "context_backend.md",
        "Backend and Core Logic Context",
        files,
        lambda f: any(
            role in f.roles
            for role in ["handler_controller", "service_logic", "data_model", "route_or_api"]
        )
        and "frontend" not in f.roles,
    )

    write_group_file(
        "context_frontend.md",
        "Frontend Context",
        files,
        lambda f: any(role in f.roles for role in ["frontend", "frontend_asset"]),
    )

    write_group_file(
        "context_tests.md",
        "Tests Context",
        files,
        lambda f: "test" in f.roles,
    )

    write_group_file(
        "context_ml_data.md",
        "ML and Data Context",
        files,
        lambda f: "ml_data" in f.roles,
    )

    write_group_file(
        "context_config_tooling.md",
        "Config and Tooling Context",
        files,
        lambda f: any(role in f.roles for role in ["config", "tooling", "docs"]),
    )

    write_keywords(files)
    write_json_files(files)

    print(f"Generated context directory: {OUTPUT_DIR}")
    print(f"Start file: {README}")
    print(f"Indexed files: {len(files)}")


if __name__ == "__main__":
    main()