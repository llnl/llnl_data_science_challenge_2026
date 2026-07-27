# Repository Guidelines

## Project Structure & Module Organization

- `src/` contains the Python implementation: `mcp_server.py` defines FastMCP tools, while `skeletonization.py` provides the underlying image-processing routine.
- `data/` holds sample CT volumes, TIFF stacks, meshes, JSON metadata, and reference images. Treat these as inputs; place generated outputs in a clearly named subdirectory rather than overwriting source data.
- `images/` and `presentation/` contain documentation assets.
- `.agents/skills/` stores project-specific skills, and `.codex/agents/` stores subagent definitions. Skills live here canonically; `.claude/skills/` holds relative symlinks to them so Claude Code discovers the same files rather than a second copy that can drift.
- `evals/` holds LLM-as-judge rubrics and the scripts that manufacture their inputs (for example `perturb_lattice_json.py`). There is no runner; rubrics are applied by hand to an agent's report.
- `.mcp.json` registers the FastMCP server for Claude Code. It invokes plain `python`, so that interpreter must be the environment where `pip install -r requirements.txt` was run — `skeleton_graph` imports `skan` at module scope, so a missing dependency takes down the whole server, not one tool.
- `README.md` and `DATA_SCIENCE_CHALLENGE_2026.pdf` describe the exercises and expected workflows.

## Setup, Test, and Development Commands

Use Python 3.11 in an isolated environment:

```bash
conda create -n dssi_env python=3.11 -y
conda activate dssi_env
pip install -r requirements.txt
python src/mcp_server.py
```

The final command starts the FastMCP server over standard I/O. For a quick syntax check, run `python -m compileall src`. Large repository assets use Git LFS; install Git LFS before cloning or checking out data files.

## Coding Style & Naming Conventions

Follow standard PEP 8 conventions: four-space indentation, `snake_case` for functions and variables, and `UPPER_CASE` for constants. Add type annotations to public functions and all MCP tool parameters because FastMCP derives schemas from them. Tool docstrings should explain inputs, outputs, file formats, and side effects. Use `pathlib.Path` or explicit path validation for new file operations. Keep scientific processing logic separate from MCP wrappers so it can be tested directly.

## Testing Guidelines

Tests live under `tests/` and run with `pytest -q`. Name files `test_<module>.py` and functions `test_<behavior>()`. Prefer small synthetic NumPy arrays over committed binary fixtures; `tests/conftest.py` builds a miniature lattice and matching volume for anything junction-related. Test success cases plus missing files, invalid axes or slice indices, unexpected dimensions, and output creation. Run future tests with `pytest -q`.

## Commit & Pull Request Guidelines

Recent history uses short, imperative summaries such as `updated presentation with github link`. Keep commits focused and write concise subjects describing the result. Pull requests should include a summary, validation commands, and any generated-output paths. Link the relevant issue or challenge task. Include before/after images when visualization behavior changes, and avoid committing generated volumes or other large binaries unless they are intentional Git LFS assets.
