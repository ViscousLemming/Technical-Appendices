# JOSS submission checklist for Stack Theory Suite

This file is not part of the JOSS paper.
It is a practical pre submission checklist aligned with the JOSS review process.

Repository and archive

- Confirm the repository is public and has a clear open source license file at the root.
- Tag a release that matches the version named in pyproject.toml.
- Create an archival release on Zenodo that mints a DOI for that tagged version.
- Add the Zenodo DOI badge and a recommended citation to the README.

Install and run

- Confirm a clean install works using pip from the tagged release.
- Confirm the minimum supported Python version is correct.
- Provide a quick start snippet that runs in under one minute on a laptop.

Documentation

- Ensure the README explains what the software does and who it is for.
- Ensure the docs include at least one end to end example that a new user can follow.
- Ensure the API docs show the core public types in Layers 1 to 5.

Tests and quality

- Ensure the test suite runs with a single command.
- Ensure the tests cover the semantic core, especially truth sets, extensions, weakness, and task correctness.
- If you can, add continuous integration so tests run on every push and pull request.

JOSS paper

- Place paper.md and paper.bib at the repository root.
- Ensure the paper includes the required JOSS sections.
- Keep the paper short and focused on what the software enables rather than a full theory exposition.
- Double check citations compile and point to real publications.
- Keep the AI usage disclosure section accurate.

Nice to have

- Add CITATION.cff for GitHub citation support.
- Add a minimal changelog for releases.
- Add a small benchmarks section or notebook showing typical runtime and scaling limits.
