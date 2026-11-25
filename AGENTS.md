# wandas-extras Repository Contract

- Provisional home for analysis functions that wandas does not have yet. Follow the rules in README.md (決めごと).
- Use `uv`. Before finishing run `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src tests`, `uv run pytest`.
- New analyses are `ExtrasFrame` methods with the kernel in `operations.py` (README 決めごと). They never modify the input frame and use the frame's channel unit/ref for dB.
- Reuse wandas for anything it already does; do not reimplement it here.
- No task-specific thresholds or reference answers from any benchmark.
