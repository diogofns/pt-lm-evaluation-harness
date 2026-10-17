# Como contribuir

```bash
uv venv && uv pip install -e ".[dev,ifeval]"
pytest                      # testes rápidos, sem GPU nem rede
```

O código fica em `src/pt_lm_eval/`, um módulo por assunto:

| Módulo | O que tem |
|---|---|
| `core.py` | tipos (itens, pedidos, respostas), `Benchmark`, protocolo `Evaluation`, registros |
| `data.py` | fonte no Hugging Face, `columns()`, few-shot, assinatura dos prompts |
| `mcq.py` | prompts em português e as avaliações `mcf`, `cf` e `gen` |
| `ifeval.py` | avaliação `ifeval` e os verificadores do Portuguese-IFEval |
| `benchmarks.py` | todos os benchmarks (cartões em `cards/`) |
| `config.py` | YAML do experimento e o plano |
| `run.py` | `Experiment`: execução, retomada, teste de sanidade, checagem de contexto |
| `client.py` | cliente OpenAI-compatível (vLLM, APIs) |
| `server.py` | vLLM gerenciado (local ou Docker) |
| `report.py` | `results.json`, estatística e CSV |
| `cli.py` | `pt-lm-eval run` e `pt-lm-eval benchmarks list/preview/check` |

## Adicionar um benchmark

1. **Escolha o dataset e fixe a revisão** (aba *Files and versions* no Hugging Face, ou
   `curl -s https://huggingface.co/api/datasets/<org>/<nome> | jq -r .sha`).
2. **Registre em `benchmarks.py`**. Cada linha do dataset chega como dict, mais `_split`,
   `_index` e `_subset`.
   - Colunas comuns: `columns(question=..., options=..., answer=..., answer_is="letter", subject=...)`.
     `options` é um caminho para uma lista (`"choices.text"`) ou uma lista de colunas;
     `answer_is` diz como ler o gabarito (`"letter"`, `"index"` ou `"text"`). Em `"auto"`,
     um gabarito que vale como letra e como texto de outra opção é erro.
   - Formato fora do padrão ou linhas a filtrar: uma função `row -> MCQItem | None`
     (`None` descarta a linha; os descartes são contados).

   `MCQItem(id, question, options, gold, subject=None, context=None)`: `id` único, 2 a 10
   opções não vazias, `gold` 0-based (`letter_index("B")` → 1), `context` mostrado antes
   da pergunta. Os exemplos few-shot são sorteados do próprio conjunto (nunca o próprio
   item, de preferência da mesma matéria); para outro split, passe `fewshot_source=`.

   As avaliações declaradas (padrão: `mcf`, `cf`, `gen`) rodam conforme o tipo do modelo:
   base faz `mcf` e `cf`; instruct e API fazem `gen`. Retire `cf` se houver alternativas
   que só fazem sentido vendo as outras ("todas as anteriores").
3. **Escreva o cartão** `cards/<nome>.md` com as seções Fonte, Licença, Variedade do
   português, Domínio, Construção, Contaminação e Mantenedor ("Não documentado" vale).
4. **Confira e trave os prompts**:
   ```bash
   pt-lm-eval benchmarks preview <nome> -n 2          # os prompts fazem sentido?
   pt-lm-eval benchmarks check <nome> --update-lock   # valida tudo e registra os prompts
   ```
   Se "chutar a letra mais comum" acertar bem mais que 1/nº de opções, diga no cartão.
   Se houver implementação de referência (lm-eval, lighteval), compare os itens.
5. **Rode uma vez de verdade** num modelo pequeno, com `limit: 50`, e leia algumas linhas
   do `*.samples.jsonl`.

### Versões

A versão identifica *exatamente* o que foi avaliado: aumente-a ao mudar a revisão do
dataset, a conversão das linhas ou os prompts. O `check` falha se um prompt mudou sem a
versão mudar, e versões diferentes nunca dividem coluna no CSV.

### Checklist de revisão

- [ ] Licença dos dados permite o uso declarado no cartão; revisão fixada.
- [ ] `benchmarks check` passa e o `benchmarks.lock.json` foi atualizado.
- [ ] Alguém fluente no domínio leu o `preview` de pelo menos 5 itens.
- [ ] Gabarito desbalanceado ou itens descartados estão explicados no cartão.

## Adicionar uma avaliação

Uma classe com `name`, `item_type`, `needs` (`{"loglikelihood"}` ou `{"generate"}`, que
decide em que cenário ela roda), `build(item, shots) -> pedidos` e
`score(item, respostas) -> Scored`, registrada com `EVALUATIONS.register(nome, instância)`
num módulo importado por `__init__.py` (veja `mcq.py`). `Scored.weights` dá pesos por
item a uma métrica; `scoring_version` faz amostras antigas serem refeitas quando a
pontuação muda.

## Estilo

`ruff format` e `ruff check` (o CI roda os dois), comentários e mensagens em português,
e testes para todo comportamento novo. Os testes `gpu`, `network` e `parity` rodam à
parte (`pytest -m network` etc.).
