<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Installation

The Stack Theory Suite is a normal Python package.
It depends on NumPy and PyTorch.

## Install from source

If you have the repository folder on disk, install it in editable mode.

```bash
pip install -e .
```

## Optional dependencies

The quality extra installs pytest.

```bash
pip install -e .[quality]
```

SymPy is optional.
If you want the SymPy bridge, install SymPy separately.

```bash
pip install sympy
```

## Verifying the install

Run the test suite.

```bash
pytest -q
```
