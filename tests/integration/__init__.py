"""Integration tests — boot a real server, run real requests.

These tests require Apple Silicon (MLX needs Metal) and download real model
weights from HuggingFace on first run (cached afterwards). They are marked
with both `smoke` and `integration` so the regular `pytest tests/` unit-test
run skips them; use `pytest tests/integration/ -m smoke` to invoke.
"""
