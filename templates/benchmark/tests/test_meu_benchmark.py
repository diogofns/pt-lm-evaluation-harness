from pt_lm_eval.testing import check_benchmark


def test_meu_benchmark():
    # Baixa o dataset, valida todos os itens, monta os prompts, confere o cartão e
    # exige que os prompts batam com a trava (rode os testes da raiz do pacote).
    check_benchmark("meu_benchmark", lock="benchmarks.lock.json")
