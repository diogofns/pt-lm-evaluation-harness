# pt-lm-eval

This Python 3.11+ harness evaluates language models on Portuguese benchmarks and English reference tasks. Work from the repository root. Use the project's `.venv` for the evaluator and local vLLM. [Installation](README.md#install) · [Configuration](docs/configuration.md)

## Where to work

- `src/pt_lm_eval/cli.py` is the CLI entry point. `experiment/` builds configurations, plans runs, checks context length, and writes results.
- `src/pt_lm_eval/benchmarks/` loads and normalizes data. The package discovers `benchmark.py` modules and registers concrete benchmarks by name. Keep each benchmark card alongside its implementation.
- `src/pt_lm_eval/evaluations/` builds requests and scores responses. `core/` defines the types shared with `inference/`, which manages endpoints and sends requests to models.
- `profiles/models/` contains model settings and `profiles/vllm.yaml` provides shared server options.
- `tests/test_contracts.py` checks registered benchmarks and evaluations. `tests/fixtures/` holds protocol regression cases.

## Evaluation changes

- Preserve pinned dataset revisions, prompt wording and whitespace, few-shot selection, and metrics unless the task calls for a protocol change.
- Bump the benchmark version before updating `benchmarks.lock.json` when the dataset revision, item conversion, or prompts change. Bump the evaluation's `scoring_version` when scoring changes so resumed runs recompute affected samples. [Benchmark and evaluation workflow](CONTRIBUTING.md#adding-a-benchmark)
- Keep module overviews in the Markdown guides. Use short `#` comments for non-obvious behavior and retain licensing notices.

## Checks

```bash
pytest
ruff check src tests
ruff format --check src tests
pt-lm-eval benchmarks list
```

The default test suite runs locally. After changing a benchmark or its prompts, run `pt-lm-eval benchmarks preview my_benchmark -n 5` and `pt-lm-eval benchmarks check my_benchmark` with its registered name. Use `pt-lm-eval run --profile qwen3.5-4b --benchmarks math_pt_br --limit 1 --dry-run` to inspect the plan and context budget. These CLI checks can load Hugging Face datasets and tokenizers. [Optional tests](CONTRIBUTING.md#optional-test-suites)
