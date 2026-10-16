# pt-lm-eval

Avaliador de LLMs em português: pequeno o bastante para ser auditado de ponta a
ponta e organizado para que qualquer pessoa adicione benchmarks, tipos de
avaliação e backends sem mexer no núcleo.

- **Modelos base** avaliados por log-likelihood (formulações MCF e CF).
- **Modelos instruct e de API** avaliados por geração, com extração da resposta.
- **vLLM gerenciado** (processo local ou Docker), endpoint existente ou API OpenAI-compatível.
- **Cada número é rastreável**: as amostras guardam prompts, logprobs e respostas.

## Cenários

Seguindo o [guia de avaliação do OpenEvals](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook):

| | Modelos base (`type: base`) | Modelos instruct (`type: instruct`/`api`) |
|---|---|---|
| Como | log-likelihood, sem chat template | geração com chat template |
| Avaliações | `cf`, `cf_pmi`, `mcf` | `gen` (extração da letra), `ifeval` (IFEval PT nativo) |
| Métrica principal | `acc_norm` (CF, normalização fixada no benchmark) ou `acc` (MCF) | `acc`; IFEval: `prompt_level_strict_acc` |
| Opções | `num_fewshot` | `thinking`, `fewshot_as_multiturn`, `system_prompt` |

Todo resultado de múltipla escolha traz `random_baseline` (Σ 1/nº de opções) para
comparar com o acerto do modelo.

## Instalação

```bash
uv venv && uv pip install -e ".[dev]"            # núcleo e testes rápidos
uv pip install -e ".[dev,ifeval]"                # + verificadores do IFEval PT (spaCy, langdetect, nltk)
uv pip install -e ".[dev,ifeval,lm-eval,gpu]"    # + testes de paridade com o lm-eval e teste F0
```

O vLLM não é dependência do pacote: instale-o no ambiente onde ele vai rodar e
aponte `server.executable` para o binário, ou use `server.mode: docker`.

## Uso

```bash
pt-lm-eval benchmarks list                         # benchmarks disponíveis
pt-lm-eval benchmarks preview energy_regulacao -n 1 # prompts, sem chamar modelo
pt-lm-eval benchmarks check energy_regulacao        # valida o dataset inteiro
pt-lm-eval run configs/exemplo.yaml --dry-run       # plano + confere se os prompts cabem no max_model_len
pt-lm-eval run configs/exemplo.yaml
```

## Configuração do experimento

Um YAML com três partes (exemplo completo em `configs/exemplo.yaml`):

| Parte | O que define |
|---|---|
| geral (topo do arquivo) | `experiment`, `seed`, `limit`, `num_fewshot`, `fewshot_strategy`, `fewshot_as_multiturn`, `server` (vLLM e `max_model_len`), `reporters` |
| `models` | cada modelo: `name`, `type` (base, instruct ou api), `thinking`, `generate` (`max_tokens`, `temperature`…), `server_args`, `system_prompt` |
| `benchmarks` | cada dataset: `name`, `num_fewshot`, `fewshot_strategy`, `fewshot_as_multiturn`, `limit` (número ou `all`) |

As avaliações não se escolhem no YAML: vêm do `type` do modelo (tabela de cenários acima)
e do que cada benchmark declara; `evaluations` num benchmark, se usado, só restringe essa lista.
O que vem no benchmark ou no modelo sobrescreve o geral; `server_args` de um modelo
se soma a `server.args`. Chaves desconhecidas são erro, para pegar erros de digitação.
Rode com `--dry-run` antes: mostra o plano e confere se os prompts cabem no contexto.

## Licença

Apache-2.0. Trechos portados do lm-evaluation-harness (MIT) mantêm o aviso original.
