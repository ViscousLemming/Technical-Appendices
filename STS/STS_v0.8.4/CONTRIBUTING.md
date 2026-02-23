<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Contributing

The full contributing guide lives in docs/contributing.md.

Quick rules

- Add tests for any semantic change
- Explain semantics in plain english
- Do not change Layer 1 meanings without updating the specification and proof obligations

Development workflow

```bash
pip install .[quality]
pytest -q
```
