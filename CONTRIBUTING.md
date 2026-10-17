# Como contribuir

```bash
uv venv && uv pip install -e ".[dev,ifeval]"
pytest                      # testes rápidos, sem GPU nem rede
```

O código fica em `src/pt_lm_eval/`, um subpacote por camada:

```
core/           itens, pedidos e respostas, Scored, Registry
benchmarks/     base.py (Benchmark, ColumnBenchmark, Columns, HFDataset)
                <nome>/benchmark.py + card.md — um benchmark por pasta
evaluations/    base.py (Evaluation) · prompts.py · multiple_choice.py (mcf, cf, gen)
                ifeval/ (InstructionFollowing + uma classe Instruction por instrução)
inference/      client.py (Client) · server.py (Endpoint → ExternalEndpoint, LocalVLLM, DockerVLLM)
experiment/     config.py (YAML e plano) · runner.py (Experiment) · lengths.py · report.py (Results)
cli.py          pt-lm-eval run | benchmarks list/preview/check
```

Benchmarks e avaliações são subclasses: ao serem definidas com um `name`, se registram
em `BENCHMARKS` ou `EVALUATIONS`. Cada pasta em `benchmarks/` com um `benchmark.py` é
importada automaticamente.

## Adicionar um benchmark

1. **Escolha o dataset e fixe a revisão** (aba *Files and versions* no Hugging Face, ou
   `curl -s https://huggingface.co/api/datasets/<org>/<nome> | jq -r .sha`).
2. **Crie a pasta `benchmarks/<nome>/` com o `benchmark.py`**. Colunas comuns bastam uma
   `ColumnBenchmark`:
   ```python
   from ..base import Columns, ColumnBenchmark, HFDataset

   class MeuBenchmark(ColumnBenchmark):
       name = "meu_benchmark"
       version = "1.0"
       description = "Uma linha sobre o que mede."
       source = HFDataset("org/dataset", split="test", revision="<commit>")
       columns = Columns(question="question", options="choices.text", answer="answer", answer_is="letter")
   ```
   `options` é um caminho para uma lista (`"choices.text"`) ou uma lista de colunas;
   `answer_is` diz como ler o gabarito (`"letter"`, `"index"` ou `"text"`; em `"auto"`, um
   gabarito que vale como letra e como texto de outra opção é erro). Formato fora do padrão
   ou linhas a filtrar: herde de `Benchmark` e implemente `to_item(row) -> MCQItem | None`
   (`None` descarta a linha). Cada linha chega como dict, mais `_split`, `_index` e `_subset`.

   `MCQItem(id, question, options, gold, subject=None, context=None)`: `id` único, 2 a 10
   opções não vazias, `gold` 0-based (`letter_index("B")` → 1), `context` mostrado antes da
   pergunta. Os exemplos few-shot são sorteados do próprio conjunto (nunca o próprio item,
   de preferência da mesma matéria); para outro split, defina `fewshot_source`.

   As avaliações declaradas (`evaluations`, padrão `mcf`, `cf`, `gen`) rodam conforme o tipo
   do modelo: base faz `mcf` e `cf`; instruct e API fazem `gen`. Retire `cf` se houver
   alternativas que só fazem sentido vendo as outras ("todas as anteriores").
3. **Escreva o cartão `card.md`** na mesma pasta, com as seções Fonte, Licença, Variedade do
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

Herde de `Evaluation` (`evaluations/base.py`) e defina `name`, `item_type`, `needs`
(`{"loglikelihood"}` ou `{"generate"}`, que decide o cenário em que ela roda),
`build(item, shots, multiturn=False)` e `score(item, respostas) -> Scored`; ao ser
definida, ela se registra. Coloque o módulo em `evaluations/` e importe-o em
`evaluations/__init__.py` (veja `multiple_choice.py`). `Scored.weights` dá pesos por item a
uma métrica; `scoring_version` faz amostras antigas serem refeitas quando a pontuação muda.

## Estilo

`ruff format` e `ruff check` (o CI roda os dois), comentários e mensagens em português,
e testes para todo comportamento novo. Os testes `gpu`, `network` e `parity` rodam à
parte (`pytest -m network` etc.).
