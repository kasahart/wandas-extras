# wandas-extras Repository Contract

- Provisional home for analysis functions that wandas does not have yet. Follow the rules in README.md (決めごと).
- Use `uv`. Before finishing run `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src tests`, `uv run pytest`.
- Functions take wandas frames, never modify them, and use the frame's channel unit/ref for dB.
- Reuse wandas for anything it already does; do not reimplement it here.
- No task-specific thresholds or reference answers from any benchmark.
