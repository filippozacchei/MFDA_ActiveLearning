#!/usr/bin/env -S python

# /// script
# dependencies = ["nox>=2025.10.16"]
# ///

import nox

nox.needs_version = ">=2025.10.14"
nox.options.default_venv_backend = "virtualenv"
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = ("3.10",)
CODE_LOCATIONS = ("src", "tests", "examples", "docs", "noxfile.py")


def install(session: nox.Session, extra: str | None = None) -> None:
    target = "."
    if extra:
        target += f"[{extra}]"
    session.install("-e", target)


@nox.session(python=PYTHON_VERSIONS, default=True)
def tests(session: nox.Session) -> None:
    install(session, "dev")
    session.run("pytest", "-q", "--tb=short")


@nox.session(python=PYTHON_VERSIONS, default=True)
def lint(session: nox.Session) -> None:
    install(session, "dev")
    session.run("ruff", "check", "--fix", *CODE_LOCATIONS)


@nox.session(python=PYTHON_VERSIONS, default=True)
def typecheck(session: nox.Session) -> None:
    install(session, "dev")
    session.run("mypy", "src")


@nox.session(python=PYTHON_VERSIONS, default=False)
def docs(session: nox.Session) -> None:
    install(session, "docs")
    session.run("mkdocs", "build", "--strict")


if __name__ == "__main__":
    nox.main()
