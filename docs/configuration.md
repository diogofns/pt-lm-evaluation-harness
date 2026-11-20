# Configuration Guide

## Model profiles

Select the model with `--profile` and pass benchmark names in `--benchmarks`. Each model has its own YAML file. Shared vLLM options come from `profiles/vllm.yaml`. [Default vLLM configuration](../profiles/vllm.yaml)

| Profile | Model source |
| --- | --- |
| [qwen3.5-4b](../profiles/models/qwen3.5-4b.yaml) | Hugging Face |
| [nemotron-3-nano-4b](../profiles/models/nemotron-3-nano-4b.yaml) | Hugging Face |
| [lumi](../profiles/models/lumi.yaml) | Local checkpoint via `--model` |
| [s5-nano](../profiles/models/s5-nano.yaml) | Local checkpoint via `--model` |

Use `--model` to supply or override the profile's model ID. Use `--model-type base` or `--model-type instruct` to override its type. Changing the type restores that type's default chat settings. `--profile` also accepts a YAML path, and `--vllm-config` accepts an alternate server YAML.

The model profile sets `type`, tokenizer, `thinking`, and `generate` options for sampling and output length. Its `server_args` override `args` in the shared vLLM YAML. `max_model_len` covers input and output together.

Quote `benchmark|shots` to set the few-shot count. A plain benchmark name uses its protocol default. `--limit` caps items per benchmark or grouped subtask. Omit it to evaluate all eligible items. Use `--experiment` to name the output directory, which defaults to the profile name.

## Evaluating checkpoints

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 pt-lm-eval run --profile qwen3.5-4b \
  --model /path/to/checkpoints --benchmarks 'mmmlu_pt_br|5' --experiment checkpoint-sweep
```

Pass a checkpoint directory or its parent. A directory containing `config.json` is treated as a single checkpoint. Otherwise, the harness takes immediate subdirectories with `config.json`, in name order. Each checkpoint uses the same profile and must be loadable by vLLM.

For multiple checkpoints, the harness starts one evaluation per GPU listed in `CUDA_VISIBLE_DEVICES` and starts the next checkpoint as soon as a GPU frees up. Set vLLM's tensor, pipeline, and data parallel sizes to `1` for this queue. Without a GPU list, checkpoints run sequentially. A single checkpoint uses the profile's normal vLLM settings.

## Models and scoring

| Model type | Supported scoring, depending on the benchmark |
| --- | --- |
| `base` | Log-likelihood and generation on supported tasks |
| `instruct` | Log-likelihood and generation, with the chat template enabled by default |
| `api` | Generation through an OpenAI-compatible endpoint |

For multiple choice, `mcf` scores option letters by log-likelihood and `cf` scores option text. Base models use log-likelihood by default and can generate on tasks whose protocols support base generation. Instruct models use `gen` when a multiple-choice benchmark supports it. Other benchmarks follow their supported protocols, such as log-likelihood for GPQA. GPQA requires access to its dataset on Hugging Face. [GPQA card](../src/pt_lm_eval/benchmarks/gpqa/card.md)

Set generation options per model under `generate`. For tasks with fixed settings, use `override_task_sampling: true` to override `temperature` and `top_p`, or `override_task_max_tokens: true` to override `max_tokens`.
