# Contributing to gp_active_mcmc

Thanks for your interest in improving `gp_active_mcmc`! This project targets
research-grade, multi-fidelity Bayesian inference workflows, so every change
helps other scientists reproduce and extend the results. The guidelines below
outline how to report issues, propose enhancements, and submit patches.

## Ground rules

- By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
- Prefer public GitHub issues for bugs/questions so others can follow along.
- Favor incremental pull requests (PRs) that keep tests/docs green and describe
  the research impact of the change.

## Ways to contribute

| Contribution type | Typical examples                                                                |
| ----------------- | ------------------------------------------------------------------------------- |
| Bug reports       | Incorrect diagnostics, sampler crashes, documentation mistakes                  |
| Feature requests  | New adaptive policies, surrogate options, HF/LF interfaces                      |
| Documentation     | Tutorials, API explanations, Navier–Stokes walkthroughs, troubleshooting guides |
| Testing & tooling | New regression tests, CI improvements, packaging fixes                          |

## Development setup

1. Ensure you have Python 3.10 (the version used in CI) and a functioning C/C++
   toolchain. For the PDE examples you will also need an MPI/FEniCSx stack;
   keep that isolated in its own environment.
2. Fork the repository and create a feature branch off `main`.
3. Create a virtual environment and install the development extras:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   ```

   This installs runtime dependencies plus `pytest`, `ruff`, `mypy`, `nox`,
   and doc tooling.

4. (Optional but recommended) Install the pre-commit hooks so formatting/lint
   fixes run locally:

   ```bash
   pre-commit install
   ```

## Quality gates

The `nox` sessions codify the supported workflows:

```bash
nox -s tests       # pytest -q --tb=short
nox -s lint        # ruff check --fix + ruff format
nox -s typecheck   # mypy src
nox -s docs        # mkdocs build --strict
```

- Keep `pytest` warnings clean (the config treats many warnings as errors).
- When touching docs/tutorials, run `mkdocs serve` locally to preview the site.
- If you alter notebooks under `docs/tutorials/`, ensure they remain executable
  in CI (MkDocs runs them with `mkdocs-jupyter`).
- For Navier–Stokes changes, please describe the external environment you used
  (MPI implementation, FEniCSx version, mesh size) so reviewers can reproduce
  results. Large data artifacts should not be committed; link to reproducible
  scripts in `examples/navier_stokes/` instead.

## Pull request checklist

Before opening a PR:

- [ ] Rebased on `main` and squashed noisy “fixup” commits where helpful.
- [ ] Tests, lint, type-checking, and docs builds pass locally (or failing jobs
      are documented in the PR).
- [ ] Added or updated unit tests/notebook snippets that cover the change.
- [ ] Updated documentation (README, tutorials, API reference) when behavior or
      user-facing parameters changed.
- [ ] Linked the relevant issue (if any) and described the motivation and
      impact on research workflows.

After submitting:

- Be responsive to reviewer feedback—short follow-up commits are preferred over
  force-pushes when possible.
- CI runs on GitHub Actions; please investigate failures and push fixes.

Thanks again for helping build reliable active-learning MCMC tooling!
