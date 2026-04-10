# Contributing

Thank you for contributing.

## Development Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt -r requirements-dev.txt`
3. Start backend:
   - `python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`
4. Start frontend:
   - `python -m streamlit run frontend/app.py`

## Quality Checks
Run before opening a PR:
1. `python -m pytest -q`
2. `python -m ruff check .`

Optional local hooks:
1. `pre-commit install`
2. `pre-commit run --all-files`

## Pull Request Guidelines
1. Keep changes scoped and focused.
2. Add/adjust tests for behavior changes.
3. Update docs when API/config behavior changes.
4. Include a short test plan in PR description.

## Branch Naming
Use one of:
- `feature/<short-topic>`
- `fix/<short-topic>`
- `docs/<short-topic>`
- `chore/<short-topic>`
