# pt-lm-eval

Avaliador de modelos de linguagem em português.

---

## Visão geral

O pt-lm-eval é um framework para avaliar modelos de linguagem em benchmarks em português,
com resultados reprodutíveis e auditáveis item a item. Junta o melhor do
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) e do
[lighteval](https://github.com/huggingface/lighteval) e segue o
[guia de avaliação do OpenEvals](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook).

**Recursos:**

- 12 benchmarks em português: múltipla escolha (ENEM, OAB, MMLU-PT, MMMLU, Global-MMLU,
  Belebele e outros) e seguimento de instruções (Portuguese-IFEval).
- Dois cenários, escolhidos pelo tipo do modelo: **base** avaliado por log-likelihood
  (formulações MCF e CF) e **instruct** avaliado por geração, com chat template.
- Modelos de raciocínio com `thinking` ligado ou desligado; few-shot como turnos de conversa.
- vLLM gerenciado pelo próprio framework (processo local ou Docker), servidor já existente
  ou API OpenAI-compatível (OpenAI, Gemini, Azure etc.).
- Reprodutibilidade: revisão fixada de cada dataset e trava de versões dos prompts
  (`benchmarks.lock.json`); um prompt não muda sem a versão do benchmark mudar.
- Auditoria: cada amostra guarda prompts, logprobs ou resposta, raciocínio e veredito.
- Execução sem supervisão: checagem de contexto antes de rodar, teste de sanidade de cada
  modelo e retomada de onde parou.
- Correção conferida contra as referências: log-likelihood idêntico ao do lm-eval pedido a
  pedido, itens idênticos aos do lm-eval e do lighteval, vereditos do IFEval idênticos aos
  do código de referência.

## Instalação

Para instalar o pacote a partir do repositório (com [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/diiogofernands/pt-lm-evaluation-harness.git
cd pt-lm-evaluation-harness
uv venv && uv pip install -e .
```

### Instalando extras

A instalação básica avalia múltipla escolha. Recursos adicionais vêm em extras:

Para o IFEval em português (spaCy, langdetect, nltk):

```bash
uv pip install -e ".[ifeval]"
```

Para desenvolvimento (testes e lint):

```bash
uv pip install -e ".[dev,ifeval]"
```

O vLLM não é dependência do pacote: instale-o no ambiente onde ele vai rodar e aponte
`server.executable` para o binário, ou use `server.mode: docker`. A tabela de todos os
extras está no fim deste documento.

## Uso básico

### Documentação

| Guia | Descrição |
|------|-----------|
| [Configuração de exemplo](./configs/exemplo.yaml) | Todas as opções do experimento, comentadas |
| [Como contribuir](./CONTRIBUTING.md) | Estrutura do código, como adicionar benchmarks e avaliações |
| [Plano](./PLANO.md) | Escopo, fases e decisões do projeto |
| Cartões dos benchmarks | `src/pt_lm_eval/benchmarks/<nome>/card.md`: fonte, licença, construção, contaminação |

Use `pt-lm-eval -h` para ver os comandos. Para listar os benchmarks:

```bash
pt-lm-eval benchmarks list
```

### Rodando um experimento

Um experimento é um arquivo YAML com três partes: a configuração geral, a lista de modelos
e a lista de benchmarks.

```yaml
experiment: meu_experimento
limit: 100                       # itens por benchmark (omita para usar todos)
server:
  args: {max_model_len: 8192, gpu_memory_utilization: 0.95}

models:
  - {name: Qwen/Qwen2.5-0.5B, type: base}
  - {name: Qwen/Qwen3-4B-Instruct-2507, type: instruct, generate: {max_tokens: 1024}}

benchmarks:
  - enem
  - {name: mmmlu_pt_br, num_fewshot: 5}
  - {name: ifeval_pt, limit: all}
```

```bash
pt-lm-eval run meu_experimento.yaml --dry-run   # plano + confere se os prompts cabem no contexto
pt-lm-eval run meu_experimento.yaml
```

Os modelos rodam um de cada vez, na ordem do YAML: o vLLM sobe, roda todos os benchmarks
daquele modelo e cai antes do próximo. O que vem num modelo ou benchmark sobrescreve a
configuração geral; chaves desconhecidas são erro, para pegar erros de digitação.

| Parte | O que define |
|-------|--------------|
| geral (topo do arquivo) | `experiment`, `output_dir`, `seed`, `limit`, `num_fewshot`, `fewshot_as_multiturn`, `server` |
| `models` | cada modelo: `name`, `type` (base, instruct ou api), `thinking`, `generate`, `server_args`, `system_prompt`, `base_url` |
| `benchmarks` | cada benchmark: `name`, `num_fewshot`, `fewshot_as_multiturn`, `limit` (número ou `all`) |

### Cenários: modelos base e instruct

As avaliações não se escolhem no YAML: vêm do `type` do modelo e do que cada benchmark
declara.

| | Modelos base (`type: base`) | Modelos instruct (`type: instruct` ou `api`) |
|---|---|---|
| Como | log-likelihood, sem chat template | geração com chat template |
| Avaliações | `mcf` (opções no prompt), `cf` (texto das opções) | `gen` (extração da letra), `ifeval` |
| Métrica principal | `acc` (MCF); `acc_norm` (CF, normalização fixada pelo benchmark) | `acc`; IFEval: `prompt_level_strict_acc` |

Todo resultado de múltipla escolha traz `random_baseline` (acerto esperado chutando) para
comparar com o do modelo. Benchmarks sem avaliação para o cenário (ex.: IFEval num modelo
base) aparecem no plano como "não se aplica".

### Benchmarks disponíveis

| Benchmark | O que mede | Itens | Fonte |
|-----------|------------|------:|-------|
| `enem` | ENEM 2022–2024, 5 alternativas | 537 | `maritaca-ai/enem` |
| `oab_exams` | Exame da OAB, 2010–2018 | 2.210 | `eduagarcia/oab_exams` |
| `mmmlu_pt_br` | MMLU traduzido profissionalmente (OpenAI) | 14.042 | `openai/MMMLU` |
| `global_mmlu_pt` | MMLU traduzido e revisado (Cohere) | 14.042 | `CohereLabs/Global-MMLU` |
| `belebele_pt` | Compreensão de leitura (FLORES) | 900 | `facebook/belebele` |
| `mmlu_pt_undergraduate` | Vestibulares e concursos, graduação | 19.319 | `mmlu-pt/mmlu-pt-undergraduate-full` |
| `mmlu_pt_undergraduate_hard` | Idem, subconjunto difícil | 11.184 | `mmlu-pt/mmlu-pt-undergraduate-hard` |
| `mmlu_pt_high_school` | Vestibulares e olimpíadas, ensino médio | 7.187 | `mmlu-pt/mmlu-pt-high-school-full` |
| `mmlu_pt_high_school_hard` | Idem, subconjunto difícil | 2.029 | `mmlu-pt/mmlu-pt-high-school-hard` |
| `mmlu_pt_sintetico` | Questões no estilo MMLU geradas por modelo | 11.791 | `cemig-ceia/benchmark_Qwen3-235B-A22B_MMLU_v2` |
| `energy_regulacao` | Regulação do setor elétrico (ANEEL) | 500 | `cemig-ceia/energy-eval` |
| `ifeval_pt` | Seguimento de instruções verificáveis | 535 | `gap-ufg/Portuguese-IFEval` |

Para ver os prompts sem chamar modelo e validar um benchmark inteiro:

```bash
pt-lm-eval benchmarks preview enem -n 2
pt-lm-eval benchmarks check enem
```

### vLLM gerenciado

Por padrão, o framework sobe um vLLM para cada modelo. As flags do `vllm serve` vão em
`server.args` (snake_case), e cada modelo pode somar as suas em `server_args`:

```yaml
server:
  mode: local                    # local (processo) ou docker
  port: 8000
  args: {max_model_len: 8192, gpu_memory_utilization: 0.95}
```

Para escolher a GPU, use `CUDA_VISIBLE_DEVICES=N pt-lm-eval run …`. No modo `docker`,
informe `server.image`; o cache do Hugging Face (`HF_HUB_CACHE`) e as GPUs visíveis são
repassados ao container.

### Modelos de raciocínio

Com `thinking`, o mesmo modelo pode ser avaliado com e sem raciocínio; cada modo vira uma
linha no relatório (`modelo:thinking` e `modelo:no-thinking`):

```yaml
models:
  - name: Qwen/Qwen3-30B-A3B
    type: instruct
    thinking: true
    server_args: {reasoning_parser: qwen3, max_model_len: 32768}
    generate: {max_tokens: 16384}
  - {name: Qwen/Qwen3-30B-A3B, type: instruct, thinking: false}
```

O raciocínio é separado da resposta antes da pontuação e fica registrado na amostra.

### APIs e servidores existentes

Modelos de API são avaliados por geração. A chave vem de uma variável de ambiente:

```yaml
models:
  - name: gpt-5.4-mini
    type: api
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    generate: {reasoning_effort: low, temperature: null}
```

Um servidor OpenAI-compatível que já está no ar é usado com `base_url`, sem subir vLLM:

```yaml
models:
  - {name: Qwen/Qwen3-4B-Instruct-2507, type: instruct, base_url: "http://127.0.0.1:8000/v1"}
```

## Dicas de uso

- **Rode `--dry-run` antes.** Além do plano, ele mede cada prompt com o tokenizer do modelo e
  aponta quantos itens não cabem no `max_model_len` (na geração, prompt + `max_tokens`). Ajuste
  `max_model_len`, `num_fewshot` ou `max_tokens` no YAML; nada é cortado sem você saber.
- **Teste de sanidade.** Assim que o modelo sobe, o framework faz um pedido de cada tipo que
  ele vai usar. Se o servidor não sobe ou o teste falha, o modelo é pulado com o motivo no
  log, os demais seguem e a execução termina com erro listando os pulados.
- **Modelos de raciocínio precisam de `max_tokens` grande.** A métrica `truncated` mostra
  quantas respostas foram cortadas pelo limite.

## Salvando resultados e retomando

Os resultados ficam em `<output_dir>/<experiment>/`:

| Arquivo | Conteúdo |
|---------|----------|
| `<modelo>/<benchmark>.<avaliação>.samples.jsonl` | uma linha por item: prompts, few-shot, logprobs ou resposta, métricas |
| `<modelo>/<benchmark>.<avaliação>.results.json` | métricas agregadas (média ± erro-padrão, por matéria), configuração e tempo |
| `<experiment>.csv` | uma linha por modelo, uma coluna por benchmark@versão\|avaliação\|métrica |
| `logs/<modelo>.vllm.log` | log do vLLM |
| `experiment.yaml` | cópia da configuração usada |

A média principal é sobre os itens, como no lm-eval; a média entre matérias fica em
`metrics_macro`.

> [!TIP]
> Uma execução interrompida retoma de onde parou: basta rodar o mesmo comando. As amostras
> só são reaproveitadas se a configuração for a mesma (modelo, benchmark e versão, prompts,
> few-shot, seed); se algo mudou, o passo recomeça.

## Contribuindo

Para começar, instale as dependências de desenvolvimento e rode os testes:

```bash
uv pip install -e ".[dev,ifeval]"
pytest                          # testes rápidos, sem GPU nem rede
pytest -m network               # baixa os datasets e confere a trava dos prompts
```

O guia [CONTRIBUTING.md](./CONTRIBUTING.md) explica a estrutura do código e como adicionar
um benchmark (uma pasta em `src/pt_lm_eval/benchmarks/` com `benchmark.py` e `card.md`) ou
uma avaliação.

### Implementando novos benchmarks

Ao decidir prompts e detalhes de avaliação, seguimos esta ordem de prioridade:

1. Se há uma implementação de referência oficial (do autor do dataset), seguimos a dela e
   comparamos item a item.
2. Se o lm-eval ou o lighteval já implementam o benchmark, conferimos a paridade dos itens
   com eles.
3. Seguimos o guia de avaliação do OpenEvals para formulação e métricas (base por
   log-likelihood, instruct por geração).
4. Prompts em português seguem o formato único de `evaluations/prompts.py`; mudar um prompt
   exige aumentar a versão do benchmark.

Não alteramos os datasets: defeitos encontrados são documentados no cartão do benchmark.

## Extras opcionais

Os extras são instalados com `uv pip install -e ".[NOME]"`.

| NOME | Descrição |
|------|-----------|
| `ifeval` | IFEval em português (spaCy com `pt_core_news_sm`, langdetect, nltk) |
| `dev` | Testes e lint (pytest, ruff) |
| `lm-eval` | lm-evaluation-harness, para gerar as referências dos testes de paridade |
| `gpu` | PyTorch, para o teste que compara o log-likelihood do vLLM com o transformers |

## Licença

Apache-2.0. Trechos portados mantêm o aviso original: a tokenização de pares, do
lm-evaluation-harness (MIT), e os verificadores do IFEval, do Portuguese-IFEval (Apache-2.0).
