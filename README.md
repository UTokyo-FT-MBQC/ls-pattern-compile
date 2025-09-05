# ls-pattern-compile
A compiler from lattice surgery commands to pattern

## Developer Guide

### Install the library

You can install this library in the editable mode via

```sh
pip install -e ./ls-pattern-compile
```

This library depends on the [`graphix-zx`](https://github.com/TeamGraphix/graphix-zx) library. If you haven't installed, you can install it via

```sh
pip install -e ./ls-pattern-compile[dev]
```

### Linter and Type Checker

We are using the following development tools

- `ruff`: ultrafast linter and formatter
- `pytest`: unittest
- `mypy`: type checker
- `pyright`: type checker
- `pre-commit`: hook management tool

For the above 4 tools, you can see the details of config in `pyproject.toml`. We require all these checkers pass before merging the Pull Request.

For `pre-commit`, please install the pre-commit config before your first commit.

```sh
pip install pre-commit
pre-commit install
```
