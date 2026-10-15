from pt_lm_eval.testing import check_benchmark


def test_meu_benchmark():
    # Baixa o dataset, valida todos os itens, monta os prompts e confere o cartão.
    check_benchmark("meu_benchmark")
