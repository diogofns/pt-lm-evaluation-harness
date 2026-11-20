# Contributing

## Development setup

Activate the virtual environment, then install the development and IFEval dependencies. [Installation guide](README.md#install)

```bash
python -m pip install -e ".[dev,ifeval]"
python -m nltk.downloader punkt_tab
pytest
ruff check src tests
ruff format --check src tests
```

### Optional test suites

Run the Hugging Face dataset tests with `pytest -m network`. For the GPU test, start vLLM with the same model that Transformers will load, then install the `gpu` extra in the evaluator environment.

```bash
python -m pip install -e ".[dev,gpu]"
PT_LM_EVAL_URL=http://127.0.0.1:8000/v1 \
PT_LM_EVAL_MODEL=Qwen/Qwen2.5-0.5B pytest -m gpu
```

For Portuguese-IFEval parity, use a checkout of the reference project at `a8d8b3c` with its published results under `experiments/`. The test skips known differences involving language detection, `OBS.` postscripts, and repeat-prompt item 50. [Portuguese-IFEval reference](https://github.com/ark-aiml-research/Portuguese-IFEval/tree/a8d8b3c)

```bash
PT_LM_EVAL_IFEVAL_REF=/path/to/Portuguese-IFEval pytest -m parity
```

## Repository layout

```text
docs/              Configuration guide
profiles/          Model profiles and shared vLLM settings
src/pt_lm_eval/
  benchmarks/     Dataset loaders and benchmark cards
  core/           Items, requests, responses, and scores
  evaluations/    Prompts and scoring
  inference/      Model clients and vLLM servers
  experiment/     Run planning, execution, and reporting
  cli.py          Command-line entry point
tests/             Test suite and fixtures
```

## Adding a benchmark

### Define the dataset and task

Pin the Hugging Face dataset to a commit SHA and create `src/pt_lm_eval/benchmarks/<name>/benchmark.py`. The package discovers these modules automatically and registers concrete benchmark classes by `name`.

For a dataset with standard multiple-choice columns, use `ColumnBenchmark`.

```python
from ..base import ColumnBenchmark, Columns, HFDataset


class MyBenchmark(ColumnBenchmark):
    name = "my_benchmark"
    version = "1.0"
    description = "A short description of what the benchmark measures."
    source = HFDataset("org/dataset", split="test", revision="<commit-sha>")
    columns = Columns(question="question", options="choices.text", answer="answer", answer_is="letter")
```

`options` accepts a path to a list or a list of column names. Set `answer_is` to `letter`, `index`, or `text` to match the dataset. For custom preprocessing, subclass `Benchmark` and implement `to_item(row)`. Return `None` to skip a row. Multiple-choice items need unique IDs and zero-based answer indices. Use `fewshot_source` when examples come from another split.

Declare the scoring methods the task supports in `evaluations`. Omit `cf` for choices that depend on the other options, such as “all of the above.” Match the reference protocol when one exists and document any deviations in the card.

### Write the benchmark card

Add `card.md` next to `benchmark.py` with the required headings `Fonte`, `Licença`, `Variedade do português`, `Domínio`, `Construção`, `Contaminação`, and `Mantenedor`. Record the dataset license, discarded rows, and substantial answer-label imbalance. [Example card](src/pt_lm_eval/benchmarks/belebele_pt/card.md)

### Validate the benchmark

Review the rendered prompts and check that the correct answers match the source data. Record the dataset revision and prompt hashes in `benchmarks.lock.json`.

```bash
pt-lm-eval benchmarks preview my_benchmark -n 5
pt-lm-eval benchmarks check my_benchmark --update-lock
pt-lm-eval benchmarks check my_benchmark
```

Bump the benchmark version before updating the lock if you change its dataset revision, item conversion, or prompts. Add local test cases for the behavior you changed. Run a small evaluation and inspect the resulting `*.samples.jsonl` before opening a pull request.

CI checks public benchmarks and previews their prompts on pull requests. To include GPQA validation on pushes to `main`, set the repository's `HF_TOKEN` Actions secret to a read token from a Hugging Face account with access to GPQA.

## Adding an evaluation

Subclass `Evaluation` and define `name`, `item_type`, and `needs` (`loglikelihood` or `generate`). Implement `build(item, shots, multiturn=False)` and `score(item, responses)`, then import the class in `evaluations/__init__.py` to register it. Use `Scored.weights` for per-item metric weights and bump `scoring_version` when scoring changes so previous samples are rescored. [Multiple-choice implementations](src/pt_lm_eval/evaluations/multiple_choice.py)
