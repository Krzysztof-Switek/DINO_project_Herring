from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
STATE_PATH = PROJECT_ROOT / "state.json"
NEXT_STEP_PATH = PROJECT_ROOT / "next_step.txt"
PROGRESS_PATH = PROJECT_ROOT / "progress.md"
LOG_PATH = PROJECT_ROOT / "controller.log"

SESSION_NAME = "otolith-dino-standalone"
SLEEP_ON_FAILURE_SECONDS = 60


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def read_next_step() -> str:
    return NEXT_STEP_PATH.read_text(encoding="utf-8").strip()


def run_claude_resume(prompt: str) -> int:
    cmd = [
        "claude",
        "--resume",
        SESSION_NAME,
        prompt,
    ]
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_claude_new(prompt: str) -> int:
    cmd = [
        "claude",
        "-n",
        SESSION_NAME,
        "--permission-mode",
        "auto",
        prompt,
    ]
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def smoke_test() -> int:
    cmd = ["python", "-m", "pytest", "-q"]
    log(f"Running smoke test: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> None:
    state = load_state()
    next_step = read_next_step()

    prompt = (
        "Read README.md, state.json, progress.md and next_step.txt first. "
        "Complete only the requested next step. "
        "Do not leave the repository scope. "
        "At the end update state.json, progress.md and next_step.txt. "
        f"Task: {next_step}"
    )

    log("Starting controller cycle")

    # First try resume, if it fails then try starting named session
    code = run_claude_resume(prompt)
    if code != 0:
        log("Resume failed, trying a new named session")
        code = run_claude_new(prompt)

    if code != 0:
        state["status"] = "paused_after_failure"
        state["last_error"] = f"Claude returned non-zero exit code: {code}"
        save_state(state)
        log("Claude failed; sleeping before next manual retry")
        time.sleep(SLEEP_ON_FAILURE_SECONDS)
        return

    test_code = smoke_test()
    if test_code == 0:
        state["status"] = "stage_completed_or_ready_for_review"
        state["last_error"] = None
    else:
        state["status"] = "tests_failed"
        state["last_error"] = f"pytest exit code: {test_code}"

    save_state(state)
    log(f"Cycle finished with pytest code {test_code}")


if __name__ == "__main__":
    main()