# Configuration Guide

Run an experiment with `pt-lm-eval run <config.yaml>`. The YAML defines an `experiment` name, the `models` and `benchmarks` to evaluate, and settings shared across the run.

## Examples

| Configuration | Use case |
| --- | --- |
| [quickstart.yaml](../examples/quickstart.yaml) | A small Qwen model on MATH-PT and Portuguese-IFEval |
| [checkpoints_multigpu.yaml](../examples/checkpoints_multigpu.yaml) | A queue of local checkpoints, one model per GPU |
| [reference_benchmarks.yaml](../examples/reference_benchmarks.yaml) | GPQA, GSM8K, MMLU-Pro, MMLU-Redux, and IFEval in English |

Set checkpoint paths, GPU IDs, and the vLLM executable to match your environment. GPQA requires access to its dataset on Hugging Face. [GPQA card](../src/pt_lm_eval/benchmarks/gpqa/card.md)

## Models and scoring

| Model type | Supported scoring, depending on the benchmark |
| --- | --- |
| `base` | Log-likelihood and generation on supported tasks |
| `instruct` | Log-likelihood and generation, with the chat template enabled by default |
| `api` | Generation through an OpenAI-compatible endpoint |

For multiple choice, `mcf` scores option letters by log-likelihood and `cf` scores option text. Set `evaluations: [gen]` for a benchmark that supports scoring generated answers.

Set generation options per model under `generate`. For tasks with fixed settings, use `override_task_sampling: true` to override `temperature` and `top_p`, or `override_task_max_tokens: true` to override `max_tokens`.

## Sampling and few-shot examples

Set `limit` to sample items and `num_fewshot` to choose how many examples to include in each prompt. Per-benchmark settings override run defaults when the protocol allows it. Use `limit: all` on a benchmark to evaluate every item. For grouped benchmarks, the limit applies per subtask.

For grouped multiple-choice benchmarks that draw examples from their own test set, `fewshot_holdout: true` reserves the first `num_fewshot` items in each group and excludes them and their duplicates from scoring.
