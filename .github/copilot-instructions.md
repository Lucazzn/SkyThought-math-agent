# SkyThought-math-agent: AI Coding Agent Instructions

## Project Architecture
- **Major Components:**
  - `skythought/evals/`: Data generation & evaluation CLI and API (`skythought evaluate`, `generate`, `score`).
  - `skythought/train/`: Training scripts, integrates [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
  - `skythought/skythought-rl/`: RL training for Sky-T1-7B, Sky-T1-mini. See `verl/` for veRL integration.
  - `recipes/`: Data curation and training strategies for Sky-T1 models.
  - `langchain-sky/`: LangChain-based integrations, tools, and conventions.

## Developer Workflows
- **Environment Setup:**
  - Use `conda create -n verl python==3.9` for RL workflows.
  - Install dependencies: `pip install -r requirements.txt`, `pip install vllm==0.6.3 ray flash-attn --no-build-isolation`.
  - For package management: `uv add <package>`, `uv sync`, `uv lock`.
- **Build/Run:**
  - RL training: `cd examples/sky-t1 && bash ./run-sky-t1-7b-zero.sh`
  - Evaluation: `skythought evaluate --model <model> --task <task> ...`
  - Data prep: `python data/data_prepare_*.py --output <path>`
- **Testing:**
  - Unit tests: `pytest tests/unit_tests/` (no network calls)
  - Integration tests: `pytest tests/integration_tests/` (network calls allowed)
  - Use `make test` for standard test runs.
- **Lint/Format:**
  - `make lint`, `make format`, type check with `uv run --group lint mypy .`

## Project-Specific Conventions
- **Python:**
  - All public functions must have type hints and Google-style docstrings (see examples in `langchain-sky/CLAUDE.md`).
  - Stable public interfaces: preserve function signatures, argument names, and argument order.
  - Use keyword-only arguments for new parameters.
  - Mark experimental features with docstring warnings (`.. warning::`).
  - Prefer dataclasses for structured data.
- **Testing:**
  - Cover new features/bugfixes with unit tests. Use fixtures/mocks for external dependencies.
  - Tests must be deterministic and fail on broken logic.
- **Security:**
  - Never use `eval()`, `exec()`, or `pickle` on user-controlled input.
  - Handle exceptions explicitly; avoid bare `except:`.
  - Clean up resources (files, sockets, threads) properly.
- **Documentation:**
  - Document all parameters, return values, and exceptions. Types go in signatures, not docstrings.
  - Focus on "why" in docstrings, not just "what".
- **Commits:**
  - Use Conventional Commits format (e.g., `feat(core): ...`, `fix(cli): ...`).

## Integration Points
- **LangChain:**
  - Use `@tool` from `langchain_core.tools` for custom tools.
  - Follow patterns in `langchain-core` for base abstractions and callbacks.
  - Implement streaming support and avoid deprecated components.
- **veRL/PRIME:**
  - RL code builds on [VeRL](https://github.com/volcengine/verl) and [PRIME](https://github.com/PRIME-RL/PRIME). See `skythought/skythought-rl/verl/` for details.
- **External Models/Data:**
  - Models and datasets are hosted on HuggingFace. Some datasets (e.g., GPQADiamond) require authentication (`huggingface-cli login`).

## Quick Reference Checklist
- [ ] No breaking public API changes
- [ ] All functions have type hints
- [ ] New code is fully tested
- [ ] No dangerous patterns (eval, silent failures)
- [ ] Docstrings for public functions
- [ ] Lint/format passes
- [ ] Commit message follows Conventional Commits

---
For more details, see:
- Main [README.md](../README.md)
- RL [skythought/skythought-rl/README.md]
- Evaluation [skythought/evals/README.md]
- LangChain [langchain-sky/CLAUDE.md]

*Please review and suggest improvements for any unclear or incomplete sections.*
