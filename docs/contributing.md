## General Guideline
Please refer to [Angular's guide](https://github.com/angular/angular/blob/22b96b96902e1a42ee8c5e807720424abad3082a/CONTRIBUTING.md) for contributing.

### Formatting
We use [Ruff](https://github.com/astral-sh/ruff) as the formatter and linter for Python, and Clang for C++.

Installing [`pre-commit`](https://pre-commit.com/) will format and lint your code before you commit:
```bash
# pip install pre-commit
pre-commit install
```
Alternatively, you can also run Ruff manually:
```bash
# pip install ruff
ruff check --fix --config pyproject.toml
ruff format --config pyproject.toml
```