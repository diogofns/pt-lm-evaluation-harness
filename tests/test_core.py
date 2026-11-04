"""Testes do núcleo que não precisam de rede nem GPU."""

from __future__ import annotations

import math

import pytest

from pt_lm_eval import BENCHMARKS, EVALUATIONS, Benchmark, ColumnBenchmark, Columns, MCQItem
from pt_lm_eval.benchmarks.base import Loaded, answer_index, letter_index
from pt_lm_eval.core import GenerateRequest, GenerateResponse, LoglikelihoodResponse, Registry, non_default
from pt_lm_eval.evaluations.multiple_choice import argmax
from pt_lm_eval.experiment import lengths, runner
from pt_lm_eval.experiment.config import parse_config
from pt_lm_eval.experiment.report import Results, aggregate, macro_mean_stderr, mean_stderr
from pt_lm_eval.experiment.runner import Experiment, SmokeTestError, smoke_test
from pt_lm_eval.inference.client import Client, continuation_logprob, truncate_left
from pt_lm_eval.inference.server import LocalVLLM, ServerError

ITEM = MCQItem(id="q1", question="Quanto é 2 + 2?", options=("3", "4", "5"), gold=1, subject="mat")


# --- tipos ----------------------------------------------------------------------


def test_mcq_item_validates_gold_and_options():
    with pytest.raises(ValueError, match="fora do intervalo"):
        MCQItem(id="x", question="?", options=("a", "b"), gold=2)
    with pytest.raises(ValueError, match="opção vazia"):
        MCQItem(id="x", question="?", options=("a", " "), gold=0)
    assert ITEM.letters == "ABC"


# --- registro -------------------------------------------------------------------


def test_registry_rejects_duplicates_and_suggests_names():
    reg: Registry[int] = Registry("coisa")
    reg.register("energia", 1)
    with pytest.raises(ValueError, match="já registrado"):
        reg.register("energia", 2)
    with pytest.raises(KeyError, match="Quis dizer: energia"):
        reg.get("enrgia")


def test_builtin_evaluations_and_benchmarks_are_registered():
    assert set(EVALUATIONS.names()) == {"mcf", "cf", "gen", "ifeval"}
    assert "energy_regulacao" in BENCHMARKS.names()


# --- dados ----------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [("B", 1), (" b ", 1), ("(C)", 2), ("A)", 0)])
def test_letter_index(value, expected):
    assert letter_index(value) == expected


def test_columns_accepts_letter_index_or_text():
    row = {"_split": "test", "_index": 7, "q": "?", "opts": {"text": ["x", "y"]}, "a": "y"}
    to_item = Columns(question="q", options="opts.text", answer="a").to_item
    item = to_item(row)
    assert (item.id, item.gold, item.options) == ("test-7", 1, ("x", "y"))
    assert to_item({**row, "a": 0}).gold == 0
    assert to_item({**row, "a": "B"}).gold == 1
    with_subject = Columns(question="q", options="opts.text", answer="a", subject="s").to_item
    assert with_subject({**row, "s": None}).subject is None  # nulo não vira a string "None"
    assert with_subject({**row, "s": " direito "}).subject == "direito"


class _Unregistered(Benchmark):
    """Benchmark de teste sem `name`: não se registra; os dados são postos à mão."""

    def to_item(self, row):
        return None


def test_select_shots_is_deterministic_and_excludes_self():
    pool = [MCQItem(id=f"p{i}", question="?", options=("a", "b"), gold=0) for i in range(20)]
    bench = _Unregistered()
    bench.data = Loaded(pool, pool, 0)
    shots = bench.select_shots(pool[0], 3, seed=1)
    assert shots == bench.select_shots(pool[0], 3, seed=1)
    assert pool[0] not in shots and len(shots) == 3


# --- tokenização e logprobs -------------------------------------------------------


def test_truncate_left_keeps_continuation():
    ids, start = truncate_left([1, 2, 3, 4], [5, 6], max_length=4)
    assert ids == [3, 4, 5, 6] and start == 2
    with pytest.raises(ValueError):
        truncate_left([1], [2, 3, 4], max_length=3)


def test_continuation_logprob_checks_token_alignment():
    ids = [10, 11, 12]
    prompt_logprobs = [
        None,
        {"11": {"logprob": -1.0, "rank": 1}},
        {"12": {"logprob": -2.0, "rank": 3}, "7": {"logprob": -0.5, "rank": 1}},
    ]
    r = continuation_logprob(prompt_logprobs, ids, start=1)
    assert (r.logprob, r.is_greedy, r.n_tokens) == (-3.0, False, 2)
    # Empate exato com o mais provável conta como guloso, mesmo com rank 2.
    tie = [None, {"11": {"logprob": -1.0, "rank": 2}, "5": {"logprob": -1.0, "rank": 1}}]
    assert continuation_logprob(tie, [10, 11], start=1).is_greedy
    with pytest.raises(KeyError):  # servidor tokenizou diferente: falha alta, nunca silenciosa
        continuation_logprob(prompt_logprobs, [10, 99, 12], start=1)


# --- métricas -------------------------------------------------------------------


def test_argmax_ties_pick_first():
    assert argmax([1.0, 3.0, 3.0]) == 1


def test_mean_stderr_matches_formula():
    mean, se = mean_stderr([1.0, 0.0, 1.0, 1.0])
    assert mean == 0.75 and se == pytest.approx(math.sqrt(0.25 / 4))
    assert math.isnan(mean_stderr([1.0])[1])
    # Grupos [1, 0] e [1]: média das médias = 0.75; variância agrupada de [1, 0, 1] = 1/3.
    mean, se = macro_mean_stderr([[1.0, 0.0], [1.0]])
    assert mean == 0.75 and se == pytest.approx(math.sqrt((1 / 3) * (1 / 2 + 1)) / 2)


def test_aggregate_headline_is_the_item_average_like_lm_eval():
    records = [{"subject": "a", "metrics": {"acc": 1.0}}] * 3 + [{"subject": "b", "metrics": {"acc": 0.0}}]
    agg = aggregate(records)
    assert agg["metrics"]["acc"] == 0.75  # média dos itens (weight_by_size do lm-eval)
    assert agg["metrics_macro"]["acc"] == 0.5  # média das matérias
    assert agg["by_subject"]["a"]["n"] == 3


def test_acc_norm_follows_the_benchmark_normalization_for_every_model():
    records = [
        {"metrics": {"acc_char": 1.0, "acc_token": 0.0}},
        {"metrics": {"acc_char": 0.0, "acc_token": 0.0}},
    ]
    by_char = aggregate(records)["metrics"]  # padrão: char, mesmo que token fosse melhor ou pior
    assert by_char["acc_norm"] == by_char["acc_char"] == 0.5
    assert by_char["acc_norm_stderr"] == by_char["acc_char_stderr"]
    assert aggregate(records, "token")["metrics"]["acc_norm"] == 0.0
    assert "acc_norm" not in aggregate([{"metrics": {"acc": 1.0}}])["metrics"]


def test_benchmark_rejects_unknown_cf_normalization():
    with pytest.raises(ValueError, match="cf_normalization"):

        class _Bad(ColumnBenchmark):
            name = "_normalizacao_invalida"
            version = "1"
            columns = Columns(question="q", options="o", answer="a")
            cf_normalization = "pmi"


# --- configuração -------------------------------------------------------------------


def test_config_rejects_unknown_keys():
    base = {"experiment": "x", "models": [{"name": "m", "type": "base"}], "benchmarks": ["energy_regulacao"]}
    assert parse_config(base).benchmarks[0].name == "energy_regulacao"
    with pytest.raises(ValueError, match="chaves desconhecidas"):
        parse_config({**base, "modles": []})
    with pytest.raises(ValueError, match="type deve ser"):
        parse_config({**base, "models": [{"name": "m", "type": "chat"}]})


@pytest.mark.parametrize("model_type,default", [("base", 64), ("instruct", 64), ("api", 8)])
@pytest.mark.parametrize(
    "global_value,model_value,expected",
    [(None, None, None), (32, None, 32), (32, 4, 4), (None, 4, 4), (1, None, 1)],
)
def test_concurrency_reaches_client_with_model_override(
    model_type, default, global_value, model_value, expected
):
    cfg = parse_config(
        {
            "experiment": "x",
            "concurrency": global_value,
            "models": [
                {"name": "m", "type": model_type, "base_url": "http://x/v1", "concurrency": model_value}
            ],
            "benchmarks": ["energy_regulacao"],
        }
    )
    step = cfg.plan()[0]
    client = Client.for_model(step.model, "http://x/v1", None)
    assert client.concurrency == (default if expected is None else expected)
    assert step.model is cfg.models[0]


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "8"])
@pytest.mark.parametrize("scope", ["global", "model"])
def test_concurrency_rejects_invalid_limits(value, scope):
    data = {
        "experiment": "x",
        "models": [{"name": "m", "type": "base"}],
        "benchmarks": ["energy_regulacao"],
    }
    target = data if scope == "global" else data["models"][0]
    target["concurrency"] = value
    with pytest.raises(ValueError, match="concurrency deve ser um inteiro positivo"):
        parse_config(data)


def test_moving_concurrency_to_global_preserves_resume_fingerprint():
    data = {
        "experiment": "x",
        "models": [{"name": "m", "type": "base", "concurrency": 8}],
        "benchmarks": ["energy_regulacao"],
    }
    old = Experiment(parse_config(data))
    inherited = {**data, "concurrency": 8, "models": [{"name": "m", "type": "base"}]}
    new = Experiment(parse_config(inherited))
    changed = Experiment(parse_config({**inherited, "concurrency": 16}))

    def fingerprint(experiment):
        step = experiment.steps[0]
        return experiment._fingerprint(step, "1.0", "prompt", EVALUATIONS.get(step.evaluation))

    assert non_default(old.steps[0].model) == non_default(new.steps[0].model)
    assert fingerprint(old) == fingerprint(new)
    assert fingerprint(old) != fingerprint(changed)


def test_http_pool_supports_concurrency_above_100():
    """O servidor só responde quando todos os 128 pedidos chegam: pool menor bloqueia."""
    import asyncio

    async def run():
        received = 0
        release = asyncio.Event()

        async def handle(reader, writer):
            nonlocal received
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                for line in headers.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        await reader.readexactly(int(line.split(b":", 1)[1]))
                received += 1
                if received == 128:
                    release.set()
                await release.wait()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        async with await asyncio.start_server(handle, "127.0.0.1", 0, backlog=256) as server:
            port = server.sockets[0].getsockname()[1]
            client = Client("m", f"http://127.0.0.1:{port}", chat=True, concurrency=128, max_retries=1)
            try:
                responses = await asyncio.wait_for(
                    asyncio.gather(*(client._post("/test", {}) for _ in range(128))), timeout=10
                )
                assert responses == [{}] * 128
                assert received == 128
            finally:
                release.set()
                await client.aclose()

    asyncio.run(run())


def test_stop_sequences_only_without_chat(monkeypatch):
    """Em chat, stops cortariam o raciocínio quando o modelo cita o enunciado."""
    import asyncio

    sent = {}

    async def fake_post(self, path, payload):
        sent[path] = payload
        key = "message" if path == "/chat/completions" else "text"
        body = {"content": "Resposta: A"} if key == "message" else "Resposta: A"
        return {"choices": [{key: body, "finish_reason": "stop"}]}

    monkeypatch.setattr(Client, "_post", fake_post)
    request = GenerateRequest(prompt="Pergunta: ?", stop=("\n\nPergunta:",))
    for chat in (True, False):
        backend = Client(model="m", base_url="http://x/v1", chat=chat)
        (response,) = asyncio.run(backend.generate([request]))
        assert response.text == "Resposta: A"
    assert "stop" not in sent["/chat/completions"]
    assert sent["/completions"]["stop"] == ["\n\nPergunta:"]


def test_thinking_flag_names_the_run_and_reaches_the_chat_template(monkeypatch):
    import asyncio

    base = {"experiment": "x", "benchmarks": ["energy_regulacao"]}
    on, off = parse_config(
        {
            **base,
            "models": [
                {"name": "Qwen/Q", "type": "instruct", "thinking": True},
                {
                    "name": "Qwen/Q",
                    "type": "instruct",
                    "thinking": False,
                    "generate": {"chat_template_kwargs": {"x": 1}},
                },
            ],
        }
    ).models
    assert (on.display_name, off.display_name) == ("Qwen/Q:thinking", "Qwen/Q:no-thinking")
    with pytest.raises(ValueError, match="reasoning_effort"):
        parse_config(
            {**base, "models": [{"name": "g", "type": "api", "base_url": "http://x", "thinking": True}]}
        )
    with pytest.raises(ValueError, match="exige chat template"):
        parse_config({**base, "models": [{"name": "m", "type": "base", "thinking": False}]})

    sent = []

    async def fake_post(self, path, payload):
        sent.append(payload)
        return {"choices": [{"message": {"content": "Resposta: A"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(Client, "_post", fake_post)
    for model in (on, off):
        backend = Client.for_model(model, "http://x/v1", None)
        asyncio.run(backend.generate([GenerateRequest(prompt="?")]))
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert sent[1]["chat_template_kwargs"] == {"x": 1, "enable_thinking": False}


def test_fewshot_as_multiturn_applies_only_to_chat_generation_with_shots():
    cfg = parse_config(
        {
            "experiment": "x",
            "fewshot_as_multiturn": True,
            "num_fewshot": 2,
            "models": [{"name": "i", "type": "instruct"}, {"name": "b", "type": "base"}],
            "benchmarks": [
                "energy_regulacao",
                {"name": "mmmlu_pt_br", "evaluations": ["gen"], "num_fewshot": 0},
            ],
        }
    )
    multiturn = {(s.model.name, s.benchmark, s.evaluation) for s in cfg.plan() if s.fewshot_as_multiturn}
    assert multiturn == {("i", "energy_regulacao", "gen")}


def test_fingerprint_ignores_fields_left_at_default():
    cfg = parse_config(
        {
            "experiment": "x",
            "models": [{"name": "m", "type": "base", "generate": {}}],
            "benchmarks": ["energy_regulacao"],
        }
    )
    assert non_default(cfg.models[0]) == {"name": "m", "type": "base"}


# --- teste de sanidade e servidor ------------------------------------------------------


class _FakeBackend:
    def __init__(self, logprob=-1.5, text="ok", error=None):
        self.logprob, self.text, self.error = logprob, text, error

    async def loglikelihood(self, requests):
        return [LoglikelihoodResponse(self.logprob, True, 2) for _ in requests]

    async def generate(self, requests):
        return [GenerateResponse(self.text, error=self.error) for _ in requests]


def test_smoke_test_checks_each_request_kind_used():
    import asyncio

    asyncio.run(smoke_test(_FakeBackend(), {"loglikelihood", "generate"}))
    with pytest.raises(SmokeTestError, match="geração falhou"):
        asyncio.run(smoke_test(_FakeBackend(text="", error="HTTP 404"), {"generate"}))
    with pytest.raises(SmokeTestError, match="geração vazia"):
        asyncio.run(smoke_test(_FakeBackend(text=""), {"generate"}))
    with pytest.raises(SmokeTestError, match="log-likelihood inválido"):
        asyncio.run(smoke_test(_FakeBackend(logprob=float("nan")), {"loglikelihood"}))
    asyncio.run(smoke_test(_FakeBackend(text="", error="x"), {"loglikelihood"}))  # só testa o que usa


def test_vllm_server_refuses_a_busy_port(tmp_path):
    import socket

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        with pytest.raises(ServerError, match="já está em uso"):
            with LocalVLLM("m", args={}, log_path=tmp_path / "x.log", port=port, executable="false"):
                pass


def test_failed_model_is_skipped_and_the_rest_still_runs(monkeypatch, tmp_path):
    from contextlib import contextmanager

    cfg = parse_config(
        {
            "experiment": "x",
            "output_dir": str(tmp_path),
            "models": [{"name": "quebrado", "type": "base"}, {"name": "ok", "type": "base"}],
            "benchmarks": [{"name": "energy_regulacao", "evaluations": ["mcf"]}],
        }
    )
    ran = []

    @contextmanager
    def fake_endpoint(model, server, log_dir):
        if model.name == "quebrado":
            raise ServerError("vLLM terminou antes de ficar pronto")
        yield "http://x/v1"

    async def fake_run_model(self, steps, client):
        ran.append(steps[0].model.name)

    monkeypatch.setattr(runner.Endpoint, "for_model", staticmethod(fake_endpoint))
    monkeypatch.setattr(Experiment, "_run_model", fake_run_model)
    with pytest.raises(RuntimeError, match=r"modelos pulados: \['quebrado'\]"):
        Experiment(cfg).run()
    assert ran == ["ok"]


# --- checagem de contexto do dry-run ------------------------------------------------


class _WordTokenizer:
    """Um token por palavra; o chat template acrescenta 3 palavras."""

    bos_token_id = None
    eos_token_id = 0

    class _Encoding(dict):  # como o BatchEncoding: aceita enc["input_ids"] e enc.input_ids
        __getattr__ = dict.__getitem__

    def __call__(self, texts, add_special_tokens=True):
        ids = [list(range(len(t.split()))) for t in ([texts] if isinstance(texts, str) else texts)]
        return self._Encoding(input_ids=ids[0] if isinstance(texts, str) else ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        return "<i> " + " ".join(m["content"] for m in messages) + " <a> <g>"


class _MemorySource:
    def __init__(self, rows):
        self.rows = rows

    def load(self):
        return [{**r, "_split": "test", "_index": i} for i, r in enumerate(self.rows)]

    def describe(self):
        return "memória"


def _memory_benchmark() -> str:
    """Benchmark em memória, com uma pergunta curta e uma longa (registrado uma vez)."""
    if "_contexto_teste" not in BENCHMARKS.names():
        rows = [
            {"q": "curta?", "o": ["a", "b"], "a": 0},
            {"q": " ".join(["longa"] * 40) + "?", "o": ["a", "b"], "a": 1},
        ]

        class _ContextoTeste(ColumnBenchmark):
            name = "_contexto_teste"
            version = "1"
            source = _MemorySource(rows)
            evaluations = ("mcf", "gen")
            columns = Columns(question="q", options="o", answer="a")

    return "_contexto_teste"


def test_dry_run_length_check_flags_generation_that_does_not_fit(monkeypatch):
    _memory_benchmark()
    monkeypatch.setattr(lengths, "_tokenizer", lambda name: _WordTokenizer())
    cfg = parse_config(
        {
            "experiment": "x",
            "server": {"args": {"max_model_len": 60}},
            "benchmarks": ["_contexto_teste"],
            "models": [
                {"name": "i", "type": "instruct", "generate": {"max_tokens": 30}},
                {"name": "b", "type": "base"},
                {"name": "g", "type": "api", "base_url": "http://x"},
            ],
        }
    )
    checks = {(c.step.model.name, c.step.evaluation): c for c in Experiment(cfg).check_lengths()}
    gen = checks[("i", "gen")]
    assert (gen.items, gen.over, gen.limit) == (2, 1, 60) and not gen.ok  # só a pergunta longa estoura
    assert checks[("b", "mcf")].over == 0 and checks[("b", "mcf")].ok
    assert checks[("g", "gen")].note == "API: sem tokenizer local" and checks[("g", "gen")].ok
    assert "1 itens não cabem" in gen.row()[-1]


def test_timing_is_recorded_and_accumulates_across_resumes(monkeypatch, tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace

    _memory_benchmark()
    cfg = parse_config(
        {
            "experiment": "t",
            "output_dir": str(tmp_path),
            "benchmarks": [{"name": "_contexto_teste", "evaluations": ["gen"]}],
            "models": [{"name": "m", "type": "instruct"}],
        }
    )
    (step,) = cfg.plan()
    clock = iter([100.0, 104.0, 200.0, 201.5])
    # Só a referência do runner: o event loop do asyncio também usa time.monotonic.
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    experiment = Experiment(cfg)
    result = asyncio.run(experiment.run_step(step, _FakeBackend(text="Resposta: A")))
    assert result["timing"] == {"seconds": 4.0, "items_run": 2, "items_resumed": 0}
    # Retomada com a mesma configuração e nada a fazer: soma o tempo anterior.
    result = asyncio.run(experiment.run_step(step, _FakeBackend(text="Resposta: A")))
    assert result["timing"] == {"seconds": 5.5, "items_run": 0, "items_resumed": 2}
    path = Results(cfg.out_dir).write_csv("t")
    assert "_contexto_teste@1|gen|duration_seconds" in path.read_text().splitlines()[0]
    assert json.loads(next(cfg.out_dir.glob("*/*.results.json")).read_text())["evaluation"]["fingerprint"]


def test_benchmark_parameters_override_the_general_ones():
    general = {"experiment": "x", "limit": 100, "num_fewshot": 3, "models": [{"name": "m", "type": "base"}]}
    cfg = parse_config(
        {
            **general,
            "benchmarks": [
                {"name": "energy_regulacao", "evaluations": ["mcf"]},
                {"name": "mmmlu_pt_br", "evaluations": ["mcf"], "limit": "all", "num_fewshot": 0},
                {"name": "ifeval_pt", "limit": 10},
            ],
        }
    )
    got = {s.benchmark: (s.limit, s.num_fewshot) for s in cfg.plan() if not s.skip_reason}
    assert got == {"energy_regulacao": (100, 3), "mmmlu_pt_br": (None, 0)}
    with pytest.raises(ValueError, match="'all'"):
        parse_config({**general, "benchmarks": [{"name": "ifeval_pt", "limit": "tudo"}]})


def test_evaluations_follow_the_model_type():
    cfg = parse_config(
        {
            "experiment": "x",
            "models": [
                {"name": "b", "type": "base"},
                {"name": "i", "type": "instruct"},
                {"name": "a", "type": "api", "base_url": "http://x"},
            ],
            "benchmarks": [
                "energy_regulacao",
                "ifeval_pt",
                {"name": "mmmlu_pt_br", "evaluations": ["cf", "gen"]},
            ],
        }
    )
    plan = {}
    for s in cfg.plan():
        plan.setdefault(s.model.name, []).append(s.skip_reason or f"{s.benchmark}/{s.evaluation}")
    assert plan["b"] == [
        "energy_regulacao/mcf",
        "energy_regulacao/cf",
        "não se aplica a modelos base",
        "mmmlu_pt_br/cf",
    ]
    assert plan["i"] == ["energy_regulacao/gen", "ifeval_pt/ifeval", "mmmlu_pt_br/gen"]
    assert plan["a"] == plan["i"]


def test_answer_letter_is_never_mistaken_for_option_text():
    """MMLU formal_logic: opções "A", "~A", "B", "~B" com gabarito "B" (a segunda)."""
    options = ["A", "~A", "B", "~B"]
    assert answer_index("B", options, "letter") == 1
    assert answer_index("~B", options, "text") == 3
    assert answer_index(2, options, "index") == 2
    with pytest.raises(ValueError, match="ambíguo"):
        answer_index("B", options)  # auto: letra e texto discordam
    assert answer_index("D", options) == 3  # auto sem conflito segue funcionando
    row = {"_split": "test", "_index": 0, "q": "?", "a": "A", "b": "~A", "c": "B", "d": "~B", "gold": "B"}
    columns = Columns(question="q", options=["a", "b", "c", "d"], answer="gold", answer_is="letter")
    assert columns.to_item(row).gold == 1


def test_exclusions_apply_before_limit_and_survive_worker_serialization(tmp_path):
    import asyncio
    import json
    from dataclasses import asdict

    data = {
        "experiment": "exclusions",
        "output_dir": str(tmp_path),
        "limit": 1,
        "models": [{"name": name, "type": "base"} for name in ("original", "checkpoint")],
        "benchmarks": [{"name": _memory_benchmark(), "exclude_ids": ["test-1"]}],
    }
    cfg = parse_config(data)
    restored = parse_config(json.loads(json.dumps(asdict(cfg))))
    experiment = Experiment(restored)
    for step in experiment.steps:
        items, _ = experiment.select(step)
        assert [item.id for item in items] == ["test-0"]
        assert step.exclude_ids == ("test-1",)
    result = asyncio.run(experiment.run_step(experiment.steps[0], _FakeBackend()))
    assert result["evaluation"]["exclude_ids"] == ["test-1"]
    assert result["benchmark"]["n_items"] == 2
    assert result["benchmark"]["n_evaluated"] == 1
    plain = Experiment(parse_config({**data, "benchmarks": [_memory_benchmark()]}))
    evaluation = EVALUATIONS.get("mcf")
    assert experiment._fingerprint(experiment.steps[0], "1", "p", evaluation) != plain._fingerprint(
        plain.steps[0], "1", "p", evaluation
    )


@pytest.mark.parametrize("excluded", ["test-1", [1], ["unknown"]])
def test_exclusions_reject_invalid_or_unknown_ids(excluded):
    with pytest.raises(ValueError, match="exclude_ids"):
        cfg = parse_config(
            {
                "experiment": "x",
                "models": [{"name": "m", "type": "base"}],
                "benchmarks": [{"name": _memory_benchmark(), "exclude_ids": excluded}],
            }
        )
        experiment = Experiment(cfg)
        experiment.select(experiment.steps[0])
