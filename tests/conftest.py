"""
Pytest configuration for loading test environment variables.

Automatically loads .env file from tests directory if it exists.
"""

import os
from pathlib import Path

_env_file = Path(__file__).parent / ".env"

if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
