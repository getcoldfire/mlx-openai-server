"""Click subgroup `coldfire-mlx-server models` — local HuggingFace cache management.

Subcommands (added in subsequent tasks):
  - models list   (Task 4)
  - models pull   (Task 5)
  - models rm     (Task 6)

Wired into the main `cli` group via `cli.add_command(models)` in app/cli.py.
"""

from __future__ import annotations

import click


@click.group(name="models")
def models() -> None:
    """Manage the local HuggingFace cache (list, pull, rm).

    Operates entirely on the local filesystem — no interaction with a
    running coldfire-mlx-server. The three commands are CLI utilities,
    not service operations.
    """
