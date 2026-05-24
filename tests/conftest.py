from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path() -> Path:
    """Workspace-local tmp_path for Windows setups with locked global temp dirs."""

    base = Path.cwd() / ".tmp_pytest_local"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        resolved_base = base.resolve()
        resolved_path = path.resolve()
        if resolved_path.exists() and resolved_base in resolved_path.parents:
            shutil.rmtree(resolved_path)
