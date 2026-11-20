from pathlib import Path

import pytest
import yaml

from pt_lm_eval.cli import main
from pt_lm_eval.experiment import Experiment, load_config
from pt_lm_eval.inference import Endpoint


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("profiles")
    (root / "models").mkdir(parents=True)
    settings = {
        "qwen3.5-4b": {
            "name": "org/qwen",
            "type": "instruct",
            "server_args": {"reasoning_parser": "qwen3"},
        },
        "lumi": {
            "type": "instruct",
            "chat_template": True,
            "thinking": False,
            "tokenizer": "org/lumi-tokenizer",
            "server_args": {"reasoning_parser": "nemotron_v3", "max_model_len": 32768},
            "generate": {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 2048},
        },
    }
    for name, config in settings.items():
        (root / "models" / f"{name}.yaml").write_text(yaml.safe_dump(config))
    (root / "vllm.yaml").write_text("args:\n  max_model_len: 8192\n")
    return root


def test_quick_run_combines_server_and_model_files(profiles, monkeypatch):
    profile_path = profiles / "models/lumi.yaml"
    original = profile_path.read_text()
    alternate = Path("custom-vllm.yaml")
    alternate.write_text("port: 8123\nargs:\n  max_model_len: 8192\n  dtype: bfloat16\n")
    target = str(Path.cwd() / "checkpoint with spaces")

    def check_execution(experiment, model, steps):
        assert model.name == target
        assert model.display_name == f"{target}:no-thinking"
        assert model.type == "instruct"
        assert model.thinking is False
        assert model.tokenizer == "org/lumi-tokenizer"
        assert model.generate == {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 2048}

        endpoint = Endpoint.for_model(model, experiment.cfg.server, experiment.cfg.out_dir / "logs")
        assert endpoint.serve_args() == [
            "serve",
            target,
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
            "--served-model-name",
            target,
            "--max-model-len",
            "32768",
            "--dtype",
            "bfloat16",
            "--reasoning-parser",
            "nemotron_v3",
        ]
        assert [(s.benchmark, s.evaluation, s.num_fewshot, s.limit) for s in steps] == [
            ("mmmlu_pt_br", "gen", 5, 10),
            ("math_pt_br", "gen", 0, 10),
        ]
        saved = load_config(Path("output/test-quick/experiment.yaml"))
        assert saved == experiment.cfg
        assert saved.plan() == steps
        return True

    monkeypatch.setattr(Experiment, "execute_model", check_execution)
    assert (
        main(
            [
                "run",
                "--profile",
                str(profile_path),
                "--vllm-config",
                str(alternate),
                "--model",
                target,
                "--benchmarks",
                "mmmlu_pt_br|5",
                "math_pt_br|0",
                "--limit",
                "10",
                "--experiment",
                "test-quick",
            ]
        )
        == 0
    )
    assert profile_path.read_text() == original


def test_quick_dry_run_uses_named_profile_and_protocol_defaults(profiles, monkeypatch):
    def check_plan(experiment):
        (model,) = experiment.cfg.models
        assert experiment.cfg.experiment == "qwen3.5-4b"
        assert model.name == "org/qwen"
        assert experiment.cfg.server.args_for(model) == {"max_model_len": 8192, "reasoning_parser": "qwen3"}
        assert [(s.benchmark, s.evaluation, s.num_fewshot) for s in experiment.steps] == [
            ("gsm8k", "exact_match", 5),
            ("ifeval_pt", "ifeval", 0),
            ("gpqa_diamond_zeroshot", "gpqa_mcf", 0),
        ]
        assert not experiment.cfg.out_dir.exists()
        return True

    monkeypatch.setattr(Experiment, "dry_run", check_plan)
    assert (
        main(
            [
                "run",
                "--profile",
                "qwen3.5-4b",
                "--benchmarks",
                "gsm8k",
                "ifeval_pt",
                "gpqa_diamond_zeroshot",
                "--dry-run",
            ]
        )
        == 0
    )


def test_quick_run_overrides_instruct_profile_for_base_checkpoint(profiles, monkeypatch):
    def check_plan(experiment):
        (model,) = experiment.cfg.models
        assert model.type == "base"
        assert model.thinking is None
        assert not model.uses_chat
        assert model.server_args["reasoning_parser"] == "nemotron_v3"
        assert [(step.evaluation, step.num_fewshot) for step in experiment.steps] == [("mcf", 5)]
        return True

    monkeypatch.setattr(Experiment, "dry_run", check_plan)
    assert (
        main(
            [
                "run",
                "--profile",
                "lumi",
                "--model",
                "/path/to/base",
                "--model-type",
                "base",
                "--benchmarks",
                "mmmlu_pt_br|5",
                "--dry-run",
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    ("profile", "benchmarks", "error"),
    [
        ("lumi", ["mmmlu_pt_br"], "informe --model"),
        ("qwen3.5-4b", ["mmmlu_pt_br|-1"], "benchmark inválido"),
        ("qwen3.5-4b", ["mmmlu_pt_br|0", "mmmlu_pt_br|5"], "benchmark 'mmmlu_pt_br' repetido"),
        ("qwen3.5-4b", ["ifeval_pt|5"], "o protocolo exige num_fewshot=0"),
    ],
)
def test_quick_run_rejects_invalid_selections(profiles, profile, benchmarks, error, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--profile", profile, "--benchmarks", *benchmarks, "--dry-run"])
    assert exc.value.code == 2
    assert error in capsys.readouterr().err


def test_run_requires_profile_and_benchmarks(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", "experiment.yaml", "--dry-run"])
    assert exc.value.code == 2
    assert "the following arguments are required: --profile, --benchmarks" in capsys.readouterr().err


@pytest.mark.parametrize("visible", [None, "0,1,2,3"])
def test_run_discovers_checkpoints_and_uses_visible_gpu_slots(profiles, monkeypatch, visible):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    checkpoints = [Path("checkpoints") / f"checkpoint-{step}" for step in range(100, 1000, 100)]
    for checkpoint in reversed(checkpoints):
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
    Path("checkpoints/logs").mkdir()
    Path("checkpoints/trainer_state.json").write_text("{}")

    def check_plan(experiment):
        assert [model.name for model in experiment.cfg.models] == [str(path) for path in checkpoints]
        assert [step.evaluation for step in experiment.steps] == ["gen"] * len(checkpoints)
        if visible is None:
            assert experiment.pool is None
        else:
            assert experiment.pool.slots == [(str(gpu), 8000 + gpu) for gpu in range(4)]
        return True

    monkeypatch.setattr(Experiment, "dry_run", check_plan)
    assert (
        main(
            [
                "run",
                "--profile",
                "qwen3.5-4b",
                "--model",
                "checkpoints",
                "--benchmarks",
                "mmmlu_pt_br|5",
                "--dry-run",
            ]
        )
        == 0
    )


def test_run_preserves_single_checkpoint_parallelism(profiles, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    model_path = "./checkpoints/checkpoint-100"
    checkpoint = Path(model_path)
    nested = checkpoint / "export"
    nested.mkdir(parents=True)
    for folder in (checkpoint, nested):
        (folder / "config.json").write_text("{}")
    (profiles / "vllm.yaml").write_text("args:\n  tensor_parallel_size: 4\n")

    def check_execution(experiment, model, steps):
        assert experiment.pool is None
        assert model.name == model_path
        assert experiment.cfg.server.args_for(model)["tensor_parallel_size"] == 4
        return True

    monkeypatch.setattr(Experiment, "execute_model", check_execution)
    assert (
        main(["run", "--profile", "qwen3.5-4b", "--model", model_path, "--benchmarks", "mmmlu_pt_br|5"]) == 0
    )


def test_run_reports_empty_checkpoint_directory(profiles, capsys):
    Path("checkpoints").mkdir()
    with pytest.raises(SystemExit) as exc:
        main(["run", "--profile", "qwen3.5-4b", "--model", "checkpoints", "--benchmarks", "mmmlu_pt_br"])
    assert exc.value.code == 2
    assert "nenhum checkpoint com config.json" in capsys.readouterr().err
