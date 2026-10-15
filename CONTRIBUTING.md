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
   - externo: copie `templates/benchmark/` inteira para um repositório novo.
2. **Edite `benchmark.py`**: nome, dataset, revisão e a conversão de linhas em itens.
   - Colunas comuns: `columns(question=..., options=..., answer=...)`.
   - Formato fora do padrão: uma função `to_item(row) -> MCQItem | None`.
     Devolva `None` para descartar linhas inválidas (ex.: alternativas vazias).
   - Fixe a revisão do dataset (`revision="<commit>"`), pegando o commit na aba
     *Files and versions* do Hugging Face.
3. **Preencha o cartão (`README.md`)**. Todas as seções são obrigatórias. O que não
   souber, declare como "Não documentado".
4. **Registre o plugin**:
   - oficial: acrescente uma linha em `[project.entry-points."pt_lm_eval.benchmarks"]`
     no `pyproject.toml` e reinstale com `uv pip install -e .`;
   - externo: ajuste o `pyproject.toml` do modelo e instale com `pip install -e .`.
5. **Confira**:
   ```bash
   pt-lm-eval benchmarks list                      # aparece?
   pt-lm-eval benchmarks preview <nome> -n 2       # os prompts fazem sentido?
   pt-lm-eval benchmarks check <nome> --update-lock   # valida tudo e registra os prompts
   ```
   O `check` carrega todos os itens, monta todos os prompts, confere o cartão e mostra
   a distribuição do gabarito. Se "chutar a letra mais comum" acertar bem mais que
   1/número de alternativas, diga isso no cartão.
6. **Rode uma vez de verdade** com `limit: 50` num modelo pequeno (veja
   `configs/exemplo.yaml`) e leia algumas linhas do `*.samples.jsonl`.
7. **Abra o PR** (oficial) com `benchmarks.lock.json` atualizado.

### Versões

A versão do benchmark identifica *exatamente* o que foi avaliado. Aumente-a sempre que
mudar a revisão do dataset, a conversão das linhas ou os prompts. O `check` compara o
hash dos prompts com o `benchmarks.lock.json` e falha se o prompt mudou sem a versão
mudar. Resultados de versões diferentes ficam em colunas diferentes no CSV.

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
