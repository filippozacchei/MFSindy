"""Automation entry points for linting, testing, and docs checks."""

from pathlib import Path

import nox

PYTHON_VERSIONS = ["3.10", "3.11"]
ENFORCED_PATHS = ["src", "examples", "docs", "noxfile.py"]

nox.options.sessions = ("lint", "tests")


def _install_dev(session: nox.Session) -> None:
    session.install("pip>=23.2")
    session.install(".[dev]")


@nox.session(python=PYTHON_VERSIONS, reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run ruff (lint) and black (format check)."""
    _install_dev(session)
    session.run("ruff", "check", *ENFORCED_PATHS)
    session.run(
        "black",
        "--check",
        *ENFORCED_PATHS,
    )


@nox.session(python="3.10", reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Run pytest if tests exist; otherwise run a smoke import."""
    _install_dev(session)
    tests_dir = Path("tests")
    if tests_dir.exists():
        session.run("pytest", "-q")
    else:
        session.log("tests/ not found. Running smoke checks instead.")
        session.run("python", "-m", "compileall", "-q", "src")
        session.run("python", "-c", "import mfsindy")
