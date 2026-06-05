"""Slow / soak tests — pre-release only, never run in CI.

These tests run for minutes-to-hours and require Apple Silicon. Invoked
manually via ``pytest tests/slow/ -m slow`` (or ``make test-soak`` once the
Makefile from Phase 8a lands). They share the session-scoped ``chat_server``
fixture from ``tests/integration/conftest.py``.
"""
