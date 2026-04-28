from pathlib import Path
import json


def test_startup_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "README.md",
        root / "requirements.txt",
        root / "state.json",
        root / "progress.md",
        root / "next_step.txt",
        root / ".claude" / "settings.json",
    ]
    for path in required:
        assert path.exists(), f"Missing required file: {path}"


def test_state_json_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert "project_name" in data
    assert "session_name" in data
    assert "current_stage" in data