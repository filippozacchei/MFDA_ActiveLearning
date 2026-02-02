import nox

python_versions = ["3.10"]


@nox.session(python=python_versions)
def tests(session):
    """Run all unit tests with pytest."""
    session.install("-r", "requirements.txt")
    session.run("pytest", "test")


@nox.session(python=python_versions)
def lint(session):
    """Run linters: black, flake8."""
    session.install("-r", "requirements-dev.txt")
    session.run("black", "--check", "code")
    session.run("flake8", "code")


@nox.session(python=python_versions)
def format(session):
    """Auto-format code with black."""
    session.install("-r", "requirements-dev.txt")
    session.run("black", "code")
