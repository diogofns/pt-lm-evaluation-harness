# Plano do pt-lm-eval

O pt-lm-eval é o nosso avaliador de LLMs em português. Ele junta o melhor do
[lighteval](https://github.com/huggingface/lighteval) e do
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) e
sucede o `light-benchmark`. A referência de método é o
[guia de avaliação do OpenEvals](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook).

## Escopo atual

- **Datasets:** só múltipla escolha e IFEval. Tarefas abertas (QA, matemática,
  redação) e LLM-as-judge ficam para depois.
- **Dois cenários**, como no guia:

| | Modelos base | Modelos instruct |
|---|---|---|
| Método | log-likelihood, sem chat template | geração com chat template |
| Formulações | CF (sinal cedo no treino), MCF (modelos mais treinados) | geração livre com extração da letra; IFEval |
| Métrica principal | `acc_norm` = normalização fixada no benchmark (padrão `acc_char`); `acc_pmi` em conhecimento difícil | `acc`, com `extracted` e `truncated` |
| Controles | `num_fewshot`, BOS | `thinking` on/off, `fewshot_as_multiturn`, `system_prompt` |

Todo resultado de múltipla escolha traz `random_baseline`, porque um número só
tem sentido acima do acaso.

## Fases

| Fase | Entrega | Situação |
|---|---|---|
| F0 | Log-likelihood do vLLM bate com o transformers | ✅ `tests/test_gpu_loglikelihood.py` |
| F1 | Núcleo: MCF/CF/geração, benchmarks em PT, vLLM gerenciado, retomada | ✅ |
| F2 | Paridade pedido a pedido com o lm-eval | ✅ `tests/test_parity_lm_eval.py` (Δ = 0) |
| F3 | Geração com modelos de raciocínio (raciocínio separado, stops só em completions) | ✅ |
| — | Pronto para contribuição: cartões, trava de prompts, contratos, CI, guia | ✅ |
| — | Ponte `lm_eval:<tarefa>` (IFEval etc.) | ✅ depois removida (F6) |
| F4 | Cenários base × instruct | em andamento |

### F4: cenários base × instruct

- [x] Ambiente reproduzível com uv: `uv pip install -e ".[dev,lm-eval,gpu]"`
- [x] `thinking: true|false` por modelo (`chat_template_kwargs.enable_thinking`);
      cada modo vira uma linha do relatório (`modelo:thinking`, `modelo:no-thinking`)
- [x] `fewshot_as_multiturn` (experimento ou benchmark): few-shot como turnos
      usuário/assistente em modelos de chat
- [x] `acc_norm` e `random_baseline` nos resultados
- [x] Avaliações escolhidas pelo `type` do modelo: base roda as de log-likelihood do benchmark,
      instruct/api as de geração; benchmark sem avaliação do cenário aparece como "não se aplica"
- [x] IFEval PT como benchmark nativo (`ifeval_pt`, dataset `gap-ufg/Portuguese-IFEval`)
- [ ] Rodar `mmlu_pt` e `mmlu_pt_sintetico` de ponta a ponta

### F5: revisão para uso como avaliador próprio (2026-10-15)

Objetivo: o pt-lm-eval roda todos os nossos evals sem depender de outro
repositório de avaliação em tempo de execução. O lm-eval fica só como
referência nos testes de paridade.

- [x] Extração na geração conforme o guia: `acc` vale só para o formato pedido
      ("Resposta: X"), normalizado
- [x] Teste de sanidade logo após o modelo subir (um pedido de cada tipo usado);
      se passar, não há checagem por pedido; se falhar, o modelo é pulado e a execução termina com erro
- [x] Porta ocupada é recusada, e o servidor precisa servir o modelo pedido
- [x] IFEval PT nativo: 31 verificadores portados, paridade exata com o código da referência
      (0 diferenças em 64.376 vereditos)
- [x] Normalização de CF fixada por benchmark (`cf_normalization`, padrão `char`): `acc_norm`
      usa a mesma para todos os modelos (antes, `acc_norm_max` escolhia por modelo)
- [x] Média principal sobre itens (micro), como o `weight_by_size` do lm-eval; a média entre
      matérias fica em `metrics_macro`
- [x] Novos benchmarks de MC em PT, com paridade de itens com a referência: `enem`, `oab_exams`
      (lighteval), `global_mmlu_pt`, `belebele_pt` (lm-eval)
- [x] Bug corrigido: gabarito em letra lido como texto de outra alternativa (`answer_is`);
      `mmmlu_pt_br` e `mmlu_pt_sintetico` sobem para 1.1
- [ ] Registrar versão do vLLM / imagem nos resultados
- [x] Execução sequencial (decisão de 2026-10-16): um modelo por vez com `gpu_memory_utilization: 0.95`,
      todos os benchmarks dele, depois o próximo; sem distribuir modelos entre GPUs
- [x] CI dos benchmarks também em `data/` e `core/`; teste de rede passa a exigir a trava
- [x] Hash dos prompts ignora campos no valor padrão (a trava estava desatualizada desde `024cfe4`)
- [x] Prompt maior que o contexto: resolvido pela configuração (`max_model_len`, `num_fewshot`,
      `max_tokens`, thinking); o `--dry-run` mede cada item com o tokenizer do modelo e aponta
      quantos não cabem, sem cortar nada
- [ ] Menor: vLLM local escutando em 127.0.0.1
- [ ] Decidir a licença (hoje Apache-2.0)

### F6: simplificação (2026-10-16)

- [x] De 68 arquivos e 3.710 linhas para 12 módulos (+ 9 cartões) e ~2.800 linhas, sem
      mudar o que é avaliado (trava dos prompts intocada, paridades mantidas)
- [x] Projeto só interno: sem entry points, `testing.py` nem `templates/`; benchmarks e
      avaliações se registram por import
- [x] Removidos: ponte com o lm-eval (o lm-eval fica só como ferramenta dos testes de
      paridade), `cf_pmi`, `acc_flex`, `fewshot_strategy first_n`, `server.kind`, `reporters`
- [x] Orientação a objetos onde há estado: `Experiment` (plano, execução, retomada,
      checagem de contexto), `Step`, `Client`, `ExperimentConfig.plan()`; funções puras para
      prompts, métricas e verificadores

### F7: orientação a objetos e organização (2026-10-17)

- [x] Subpacotes por camada: `core/`, `benchmarks/`, `evaluations/`, `inference/`, `experiment/`
- [x] Um benchmark por pasta (`benchmark.py` + `card.md`), cada um uma subclasse de
      `Benchmark` (ou `ColumnBenchmark`), registrada ao ser definida e importada sozinha
- [x] `Evaluation` abstrata; `LoglikelihoodMCQ` → `MultipleChoiceFormulation`, `ClozeFormulation`;
      `GenerativeMCQ`; `InstructionFollowing`; os 31 verificadores do IFEval são classes
      `Instruction`, como na referência
- [x] `Endpoint` → `ExternalEndpoint`, `VLLMServer` → `LocalVLLM`, `DockerVLLM`; `Results`
      para os arquivos do experimento; `TokenCounter` para a checagem de contexto
- [x] Comportamento idêntico: trava de prompts intocada e paridades mantidas

### F8: fila de checkpoints por GPU (2026-10-17)

- [x] `gpus` opcional: um checkpoint Hugging Face por GPU, com fila dinâmica
- [x] Processos e portas separados; GPU reutilizada após encerrar o vLLM
- [x] Falhas isoladas por modelo, cancelamento com limpeza e CSV pelo coordenador
- [x] Retomada compatível e execução sequencial mantida quando `gpus` é omitido
- [x] Testes com subprocessos simulados, sem carregar modelos nas GPUs

A opção explícita de fila amplia a decisão sequencial da F5; o padrão permanece sequencial.

## Problemas conhecidos dos datasets (fora do escopo)

Não mexemos nos datasets; o foco é o framework avaliar corretamente. Registrados para
quem for interpretar os números:

- MMLU-PT (graduação): o item `stem-1605` traz imagens base64 nas alternativas (~63 mil
  tokens; o `--dry-run` aponta) e ~3% dos itens repetem a letra nas alternativas ("A. A) …").
- Portuguese-IFEval: no item 50, o `prompt_to_repeat` difere do prompt enviado, e quem
  repete o prompt corretamente é reprovado nessa instrução.

## Depois do escopo atual

Tarefas abertas com F1/exact match; métricas de amostragem (avg@n, maj@n) para
pós-treino; viés de ordem das opções (embaralhar e repetir); relatório além do CSV.
