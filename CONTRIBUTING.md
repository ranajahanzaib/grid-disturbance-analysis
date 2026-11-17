# Contributing to `grid-disturbance-analysis`

Thank you for your interest in contributing to `grid-disturbance-analysis`! We welcome contributions from power systems researchers, signal processing engineers, machine learning practitioners, and open-source developers.

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](./CODE_OF_CONDUCT.md).

---

## Development Setup

### 1. Prerequisites

- Python 3.11 or higher
- Git
- [`uv`](https://docs.astral.sh/uv/) (recommended for fast, deterministic dependency resolution) or standard `venv`

### 2. Fork and Clone

```bash
git clone https://github.com/ranajahanzaib/grid-disturbance-analysis.git
cd grid-disturbance-analysis
```

### 3. Create Virtual Environment

Using `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or using standard `python`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## Development Workflow

### 1. Create a Topic Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Code Standards

The repository adheres to strict automated code quality standards:

- **Linting & Formatting:** Managed with `ruff`.
  ```bash
  ruff check src tests scripts examples
  ruff format src tests scripts examples
  ```
- **Type Checking:** Managed with `mypy`.
  ```bash
  mypy src/grid_disturbance
  ```
- **Tests & Coverage:** Verified with `pytest`.
  ```bash
  pytest
  ```

### 3. Pre-Commit Hooks

Install pre-commit hooks to automatically check your code prior to committing:

```bash
pre-commit install
```

---

## Pull Request Guidelines

1. **Focused Changes:** Keep pull requests focused on a single feature, bug fix, or documentation improvement.
2. **Test Coverage:** Any new signal processing algorithm, diagnostic rule, or ML model must include unit tests in `tests/`.
3. **No Breaking Scientific Changes:** Algorithms must not silently alter physical interpretations or relax causation boundaries without documented justification.
4. **Documentation:** Update relevant documents in `docs/` and docstrings when modifying public APIs.

---

## Reporting Issues

If you find a bug, encounter unexpected numerical behavior, or have a feature proposal:

- Use our [Issue Templates](https://github.com/ranajahanzaib/grid-disturbance-analysis/issues/new/choose).
- Provide minimal reproducible examples with sampling frequencies, duration, and signal data snippets where possible.
- For security vulnerabilities, review our [Security Policy](./.github/SECURITY.md).
