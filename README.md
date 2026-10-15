# pt-lm-eval

Avaliador de LLMs em português: pequeno o bastante para ser auditado de ponta a
ponta e organizado para que qualquer pessoa adicione benchmarks, tipos de
avaliação e backends sem mexer no núcleo.

- **Modelos base** avaliados por log-likelihood (formulações MCF e CF).
- **Modelos instruct e de API** avaliados por geração, com extração da resposta.
- **vLLM gerenciado** (processo local ou Docker), endpoint existente ou API OpenAI-compatível.
- **Cada número é rastreável**: as amostras guardam prompts, logprobs e respostas.

## Instalação

```bash
uv venv && uv pip install -e ".[dev]"
```

O vLLM não é dependência do pacote: instale-o no ambiente onde ele vai rodar e
aponte `server.executable` para o binário, ou use `server.mode: docker`.

## Uso

```bash
pt-lm-eval benchmarks list                         # benchmarks disponíveis
pt-lm-eval benchmarks preview energy_regulacao -n 1 # prompts, sem chamar modelo
pt-lm-eval benchmarks check energy_regulacao        # valida o dataset inteiro
pt-lm-eval run configs/exemplo.yaml --dry-run       # mostra o plano
pt-lm-eval run configs/exemplo.yaml
```

Veja `configs/exemplo.yaml` para todas as opções do experimento.

## Licença

Apache-2.0. Trechos portados do lm-evaluation-harness (MIT) mantêm o aviso original.
