"""
Shim module. The real config lives in app/config.py.
This file exists so that lib/ modules (which do `import config`)
can find it when running under the FastAPI app.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.config import *  # noqa: F401, F403
from app.config import (
    DEFAULTS,
    read_settings,
    save_settings,
    update_setting,
    get_setting,
)
