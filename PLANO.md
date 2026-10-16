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
| Métrica principal | `acc_norm_max` = máx(`acc_char`, `acc_token`); `acc_pmi` em conhecimento difícil | `acc`, com `extracted` e `truncated` |
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
| — | Ponte `lm_eval:<tarefa>` (IFEval etc.) | ✅ |
| F4 | Cenários base × instruct | em andamento |

### F4: cenários base × instruct

- [x] Ambiente reproduzível com uv: `uv pip install -e ".[dev,lm-eval,gpu]"`
- [x] `thinking: true|false` por modelo (`chat_template_kwargs.enable_thinking`);
      cada modo vira uma linha do relatório (`modelo:thinking`, `modelo:no-thinking`)
- [x] `fewshot_as_multiturn` (experimento ou benchmark): few-shot como turnos
      usuário/assistente em modelos de chat; também repassado à ponte com o lm-eval
- [x] `acc_norm_max` e `random_baseline` nos resultados
- [x] IFEval PT como benchmark nativo (`ifeval_pt`, dataset `gap-ufg/Portuguese-IFEval`)
- [ ] Rodar `mmlu_pt` e `mmlu_pt_sintetico` de ponta a ponta

### F5: revisão para uso como avaliador próprio (2026-10-15)

Objetivo: o pt-lm-eval roda todos os nossos evals sem depender de outro
repositório de avaliação em tempo de execução. O lm-eval fica só como
referência nos testes de paridade.

- [x] Extração na geração conforme o guia: `acc` vale só para o formato pedido
      ("Resposta: X"), normalizado; `acc_flex` (auxiliar) aceita menções fora do formato
- [x] Teste de sanidade logo após o modelo subir (um pedido de cada tipo usado);
      se passar, não há checagem por pedido; se falhar, o modelo é pulado e a execução termina com erro
- [x] Porta ocupada é recusada, e o servidor precisa servir o modelo pedido
- [x] IFEval PT nativo: 31 verificadores portados, paridade exata com o código da referência
      (0 diferenças em 64.376 vereditos); a ponte `lm_eval:` deixa de ser necessária para o IFEval
- [ ] Normalização de CF fixada por benchmark (hoje `acc_norm_max` escolhe por modelo)
- [ ] Registrar versão do vLLM / imagem nos resultados
- [ ] Distribuir modelos entre GPUs livres; reaproveitar o servidor entre entradas do mesmo modelo
- [x] CI dos benchmarks também em `data/` e `core/`; teste de rede passa a exigir a trava
- [x] Hash dos prompts ignora campos no valor padrão (a trava estava desatualizada desde `024cfe4`)
- [x] Prompt maior que o contexto: resolvido pela configuração (`max_model_len`, `num_fewshot`,
      `max_tokens`, thinking); o `--dry-run` mede cada item com o tokenizer do modelo e aponta
      quantos não cabem, sem cortar nada
- [ ] MMLU-PT (graduação): 1 item com imagens base64 nas alternativas (`stem-1605`, ~63 mil
      tokens) e ~3% dos itens com a letra repetida nas alternativas ("A. A) …"); limpar exige
      subir a versão do benchmark
- [x] Tempo de cada avaliação no `results.json` (`timing`) e no CSV (`duration_seconds`),
      somado entre retomadas da mesma configuração
- [ ] Menor: vLLM local escutando em 127.0.0.1
- [ ] Decidir a licença (hoje Apache-2.0)

## Depois do escopo atual

Tarefas abertas com F1/exact match; métricas de amostragem (avg@n, maj@n) para
pós-treino; viés de ordem das opções (embaralhar e repetir); relatório além do CSV.
