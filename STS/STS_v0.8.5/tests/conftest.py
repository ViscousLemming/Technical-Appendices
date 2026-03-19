# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration.

This test suite is designed to check semantic soundness.
It aims to ensure the implementation matches the Stack Theory definitions.

The only configuration we need is to ensure the repository root is on sys.path.
That lets pytest import the stacktheory package from source.
"""

import sys
from pathlib import Path

# Ensure the repository root is on sys.path for tests.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
