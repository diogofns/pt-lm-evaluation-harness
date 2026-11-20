# Portuguese Language Model Evaluation Harness

## Overview

`pt-lm-eval` evaluates language models on Portuguese benchmarks including Belebele, MMLU-PT, and Portuguese-IFEval. It supports multiple-choice scoring, text generation, and instruction following with vLLM or an OpenAI-compatible API. English reference tasks are also available.

## Install

From the repository root, create a virtual environment with Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### Instruction following

Install the `ifeval` extra for Portuguese or English IFEval, including the quickstart below. The extra installs the Portuguese spaCy model.

```bash
python -m pip install -e ".[ifeval]"
python -m nltk.downloader punkt_tab
```

### vLLM in a separate environment

For Linux with an NVIDIA GPU, keep the evaluator's `.venv` active and use `uv` to install vLLM in a separate environment. If you need `uv`, install it with `python -m pip install uv`.

```bash
vllm_venv="$HOME/.venvs/pt-lm-eval-vllm"
uv venv --python 3.12 "$vllm_venv"
uv pip install --python "$vllm_venv/bin/python" vllm --torch-backend=auto
"$vllm_venv/bin/vllm" --version
printf '%s\n' "$vllm_venv/bin/vllm"
```

Copy the absolute path printed above into `server.executable` in `examples/quickstart.yaml`. Follow the platform-specific instructions for other GPUs or a specific CUDA version. [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)

### Docker and existing endpoints

To launch vLLM in Docker, set `server.mode: docker` and `server.image`.

To use a running server, set the model's `base_url` to its API root, such as `http://127.0.0.1:8000/v1` for vLLM. Set `api_key_env` to the name of the variable holding the API key. The official OpenAI endpoint uses `OPENAI_API_KEY` by default.

## Basic Usage

List benchmarks, preview a prompt, and run the quickstart with a small Qwen model. The example evaluates MATH-PT and Portuguese-IFEval.

```bash
pt-lm-eval benchmarks list
pt-lm-eval benchmarks preview math_pt_br -e gen -n 1
pt-lm-eval run examples/quickstart.yaml --dry-run
pt-lm-eval run examples/quickstart.yaml
```

The dry run shows the evaluation plan and checks context lengths when `max_model_len` and a tokenizer are available. The example downloads its datasets and tokenizer from Hugging Face on first use.

Use `pt-lm-eval --help` for available commands. The configuration guide covers model types, scoring methods, and examples for multi-GPU runs and English reference tasks. [Configuration guide](docs/configuration.md)

## Tuning vLLM Inference

Update `server.args` in your experiment YAML. Use the vLLM flag names with underscores, such as `max_num_seqs` for `--max-num-seqs`. Set `server_args` on a model to override shared options, or pass the flags directly to `vllm serve` when using an existing server.

```yaml
concurrency: 64
server:
  executable: /home/your-user/.venvs/pt-lm-eval-vllm/bin/vllm
  args:
    max_model_len: 4096
    max_num_seqs: 64
    max_num_batched_tokens: 8192
    gpu_memory_utilization: 0.85
    enable_chunked_prefill: true
    enable_prefix_caching: true
```

| vLLM option | What to tune |
| --- | --- |
| `max_num_seqs` | Maximum sequences per scheduler step. Try raising it for throughput or lowering it when the KV cache is under pressure. |
| `max_num_batched_tokens` | Token budget per scheduler step. With chunked prefill, larger budgets can improve throughput while smaller budgets can reduce the time between generated tokens. |
| `gpu_memory_utilization` | GPU memory budget. Raise it when unused VRAM can give the KV cache more room and reduce preemptions. |
| `max_model_len` | Total context per request. Keep enough room for the prompt and generated tokens, and check it with `--dry-run`. |
| `enable_chunked_prefill` | Splits long prompts into chunks so the scheduler can interleave them with generation. |
| `enable_prefix_caching` | Reuses cached KV states for identical prompt prefixes to reduce prompt processing time. |
| `tensor_parallel_size` | Shards one model across GPUs. More shards can free memory per GPU and add synchronization overhead. Use `1` with the harness's `gpus` queue. |
| `dtype`, `quantization` | Change precision or use a supported quantized checkpoint. Measure scores as well as speed when comparing precision settings. |
| `enforce_eager` | Disables CUDA graphs and compilation when set to `true`. Useful for startup troubleshooting and typically slower for sustained inference. |

The harness passes `true` as a flag and omits `false`. Use the explicit disable flag listed by your installed `vllm serve --help` and set its YAML key to `true`. [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/) · [vLLM optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization/)

`concurrency` controls how many requests the harness sends at once. Start with the example values, then vary one setting at a time. Keep the model, sample, and generation settings fixed and use a new `experiment` name for each trial so it runs fresh. Compare `duration_seconds` in the results CSV.

## Saving and Resuming Results

Runs write to `output/<experiment>/` by default. The CSV compares models, `*.results.json` stores metrics, and `*.samples.jsonl` records prompts, responses, and scores per item. Run the same command with the same configuration to resume an interrupted run.

## Contributing

See the contribution guide for the package layout, development setup, and steps for adding and validating a benchmark. [Contributing guide](CONTRIBUTING.md)

Open the repository root in Codex or Claude Code to load the shared project guide. [AGENTS.md](AGENTS.md)

## License

Licensed under Apache-2.0. [License](LICENSE) · [Third-party notices](src/pt_lm_eval/NOTICE)
