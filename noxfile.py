import nox

PYTHON_VERSIONS = ("3.10",)
CODE_LOCATIONS = ("src", "tests", "examples", "docs", "noxfile.py")

nox.options.sessions = ("lint", "tests", "typecheck", "docs")
nox.options.reuse_existing_virtualenvs = True


def install(session, extra: str | None = None) -> None:
    """Install the project in editable mode with an optional extra."""
    target = "."
    if extra:
        target += f"[{extra}]"
    session.install("-e", target)


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the pytest suite with the dev extra."""
    install(session, "dev")
    session.run("pytest", "-q", "--strict-markers", "--tb=short")


@nox.session(python=PYTHON_VERSIONS)
def lint(session: nox.Session) -> None:
    """Run formatting and style checks."""
    install(session, "dev")
    session.run("ruff", "check", *CODE_LOCATIONS)
    session.run("black", "--check", *CODE_LOCATIONS)


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session: nox.Session) -> None:
    """Run mypy on the source tree."""
    install(session, "dev")
    session.run("mypy", "src")


@nox.session(python=PYTHON_VERSIONS)
def docs(session: nox.Session) -> None:
    """Build the MkDocs site (executes notebooks)."""
    install(session, "docs")
    session.run("mkdocs", "build", "--strict")
