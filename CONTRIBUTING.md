# Como contribuir

O pt-lm-eval cresce por plugins: benchmarks, avaliações, backends e relatórios
entram todos do mesmo jeito, implementando um contrato pequeno e registrando com um
nome. Nada no núcleo precisa mudar para adicionar um deles.

Antes de começar:

```bash
git clone <repositório> && cd pt-lm-eval
uv venv && uv pip install -e ".[dev]"
pytest                      # testes rápidos, sem GPU nem rede
```

Plugins importam apenas de `pt_lm_eval` (o `__init__.py`); o resto do pacote pode
mudar entre versões.

## 1. Adicionar um benchmark de múltipla escolha

Há dois caminhos. O código é o mesmo; muda onde ele mora.

| | No repositório principal (oficial) | No seu próprio pacote (externo) |
|---|---|---|
| Quando | Benchmark de interesse geral, com licença que permite redistribuir | Dados restritos, em desenvolvimento ou de interesse de um grupo |
| Como entra | Pull request, com revisão | `pip install` do seu pacote |
| Aparece em `benchmarks list` como | `oficial` | `externo (nome-do-pacote)` |

### Passo a passo

1. **Copie o modelo**:
   - oficial: `cp -r templates/benchmark/meu_benchmark src/pt_lm_eval/benchmarks/<nome>`
   - externo: copie `templates/benchmark/` inteira para um repositório novo (o
     comentário no topo do `pyproject.toml` pode ser apagado).

   Depois troque `meu_benchmark` pelo seu nome em todos estes lugares: a pasta, o
   nome em `BENCHMARKS.register(...)`, o título do cartão e, no pacote externo, a
   chave e o módulo do entry point, `packages = [...]`, o `name` do projeto, o nome
   do arquivo de teste e o argumento de `check_benchmark(...)`. O nome que vale para
   o pt-lm-eval é o de `BENCHMARKS.register`; a chave do entry point é só um rótulo,
   mas use o mesmo nome para não confundir.
2. **Escolha o dataset e fixe a revisão**. Use o split que contém as questões de
   avaliação; se o dataset só publica `train`, tudo bem, mas diga isso no cartão. O
   commit (`revision="..."`) está na aba *Files and versions* do Hugging Face, ou em
   `curl -s https://huggingface.co/api/datasets/<org>/<nome> | jq -r .sha`.
3. **Converta as linhas em itens** em `benchmark.py`. Cada `row` é a linha do dataset
   como dict (campos aninhados viram dicts, ex.: `row["choices"]["text"]`), mais
   `_split` (nome do split) e `_index` (posição no split).
   - **Colunas comuns**: `columns(question=..., options=..., answer=..., subject=...)`.
     `options` é um caminho para uma lista (`"choices.text"`) ou uma lista de colunas
     (`["A", "B", "C", "D"]`); `answer` aceita letra, índice 0-based ou o texto da
     opção. Sem `id=`, o ID é `"<split>-<índice>"`. `subject` nulo ou vazio vira "sem
     matéria".
   - **Formato fora do padrão ou linhas a filtrar**: uma função
     `to_item(row) -> MCQItem | None`. Devolva `None` para descartar a linha (ex.:
     alternativas vazias, questão anulada); os descartes são contados e relatados.

   `MCQItem(id, question, options, gold, subject=None, context=None)`:

   | Campo | Regra |
   |---|---|
   | `id` | texto único dentro do benchmark (repetido = erro no `check`) |
   | `question` | enunciado |
   | `options` | 2 a 10 alternativas, nenhuma vazia |
   | `gold` | índice 0-based da correta; `letter_index("B")` → 1 (aceita "b", "(B)", "B)"; outra coisa é erro) |
   | `subject` | matéria, para resultados por matéria (opcional) |
   | `context` | texto de apoio mostrado antes da pergunta (opcional) |

   **Few-shot**: por padrão os exemplos são sorteados do próprio conjunto, nunca o
   próprio item, de preferência da mesma matéria, de forma determinística. Para usar
   outro split ou dataset, passe `fewshot_source=HFDataset(...)` no `Benchmark`.
4. **Escolha as avaliações** (`evaluations=`):
   - `mcf`: as alternativas aparecem no prompt e o modelo escolhe a letra (log-likelihood);
   - `cf`: as alternativas não aparecem e o modelo pontua o texto de cada uma.
     Retire-a quando houver alternativas que só fazem sentido vendo as outras
     ("todas as anteriores", "nenhuma das alternativas");
   - `gen`: o modelo escreve a resposta e a letra é extraída (único modo para APIs).
5. **Preencha o cartão (`README.md`)**. Todas as seções são obrigatórias; o que não
   souber, declare como "Não documentado". Em *Mantenedor*, ponha quem mantém o
   benchmark no pt-lm-eval; os autores do dataset vão em *Fonte*.
6. **Instale**:
   - oficial: acrescente o módulo em `[project.entry-points."pt_lm_eval.benchmarks"]`
     no `pyproject.toml` e reinstale com `uv pip install -e .`;
   - externo: no ambiente onde está o pt-lm-eval,
     `uv pip install --python <venv>/bin/python -e .` (ou `pip install -e .` se o venv tiver pip).
7. **Confira**:
   ```bash
   pt-lm-eval benchmarks list                         # aparece? (externo: "externo (seu-pacote)")
   pt-lm-eval benchmarks preview <nome> -n 2          # os prompts fazem sentido?
   pt-lm-eval benchmarks check <nome> --update-lock   # valida tudo e registra os prompts
   pytest                                             # o teste do modelo usa a mesma trava
   ```
   O `check` carrega todos os itens, monta todos os prompts, confere o cartão e mostra
   a distribuição do gabarito. Se "chutar a letra mais comum" acertar bem mais que
   1/número de alternativas, diga isso no cartão. A trava é o arquivo
   `benchmarks.lock.json` **do diretório atual** (ou `--lock caminho`): no pacote
   externo, mantenha-a na raiz do seu repositório e rode os comandos de lá.
8. **Rode uma vez de verdade** num modelo pequeno, com poucos itens, e leia algumas
   linhas do `*.samples.jsonl`:
   ```yaml
   # teste.yaml — pt-lm-eval run teste.yaml
   experiment: teste
   limit: 50
   server: {mode: local, args: {max_model_len: 4096}}
   benchmarks: [<nome>]
   models:
     - {name: Qwen/Qwen2.5-0.5B, type: base}
   ```
9. **Abra o PR** (oficial) com o `benchmarks.lock.json` atualizado.

### Versões

A versão (texto livre, ex.: `1.0` → `1.1`) identifica *exatamente* o que foi
avaliado. Aumente-a sempre que mudar a revisão do dataset, a conversão das linhas ou
os prompts. O `check` compara o hash dos prompts com a trava e falha se um prompt
mudou sem a versão mudar; espaços nas pontas dos textos são normalizados, então só
mudanças visíveis contam. Resultados de versões diferentes ficam em colunas
diferentes no CSV e nunca se misturam.

### Checklist de revisão (benchmarks oficiais)

- [ ] Licença dos dados permite o uso declarado no cartão.
- [ ] Cartão completo, com construção e riscos de contaminação descritos.
- [ ] Revisão do dataset fixada.
- [ ] `benchmarks check` passa e o `benchmarks.lock.json` foi atualizado.
- [ ] Alguém fluente no domínio leu o `preview` de pelo menos 5 itens.
- [ ] Gabarito desbalanceado ou itens descartados estão explicados no cartão.

## 2. Adicionar um tipo de avaliação

Uma avaliação define *como* um item vira pedidos ao modelo e como as respostas viram
métricas. Exemplo completo, para QA aberta:

```python
from dataclasses import dataclass
from pt_lm_eval import EVALUATIONS, GenerateRequest, Item, Scored

@dataclass(frozen=True)
class QAItem(Item):
    question: str
    answers: tuple[str, ...]

class QAExactMatch:
    name = "qa_em"
    item_type = QAItem
    needs = frozenset({"generate"})          # tipos de pedido que usa

    def build(self, item, shots):
        return [GenerateRequest(prompt=f"Pergunta: {item.question}\nResposta:", max_tokens=64)]

    def score(self, item, responses):
        pred = responses[0].text.strip().lower()
        hit = float(any(pred == a.lower() for a in item.answers))
        return Scored(metrics={"exact_match": hit}, details={"pred": pred})

EVALUATIONS.register("qa_em", QAExactMatch())
```

Registre o módulo em `[project.entry-points."pt_lm_eval.evaluations"]`. O runner
cruza `needs` com o que cada backend oferece; uma avaliação que pede `loglikelihood`
é pulada automaticamente para modelos de API, com o motivo no plano.

Teste com o contrato pronto:

```python
from pt_lm_eval.testing import check_evaluation
check_evaluation("qa_em", QAItem(id="1", question="Capital do Brasil?", answers=("Brasília",)))
```

## 3. Adicionar um backend

Um backend responde a pedidos. Implemente `capabilities` (os tipos de pedido que
atende), `loglikelihood`, `generate` e `aclose`, e registre uma *fábrica* com
`capabilities(model)` e `create(model, base_url, max_length)` em
`[project.entry-points."pt_lm_eval.backends"]`. O modelo escolhe o backend com
`backend: <nome>` no YAML. Veja `backends/openai_compat.py` como referência.

## Estilo

- Código e mensagens em português; nomes de identificadores em inglês.
- Funções pequenas e dados explícitos; sem estado global.
- Nenhum módulo do núcleo deve passar de ~150 linhas.
- `ruff format && ruff check && pytest` antes do PR.
