"""Fila multi-GPU sem GPU: subprocessos reais verificam isolamento e ordem dinâmica."""

import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pt_lm_eval.experiment import Experiment
from pt_lm_eval.experiment.config import (
    BenchmarkRun,
    ExperimentConfig,
    ModelConfig,
    ServerConfig,
    parse_config,
)
from pt_lm_eval.experiment.pool import GPUPool
from pt_lm_eval.experiment.report import Results
from pt_lm_eval.experiment.worker import main as worker_main
from pt_lm_eval.inference.server import DockerVLLM, Endpoint


def config(tmp_path, **kwargs):
    return ExperimentConfig(
        experiment="queue",
        output_dir=str(tmp_path),
        models=tuple(ModelConfig(name, "base") for name in ("a", "b", "c")),
        benchmarks=(BenchmarkRun("enem"),),
        gpus=("0", "1"),
        **kwargs,
    )


@pytest.mark.parametrize("gpus", [[], "0,1", [True], [-1], ["bad"], ["0,1"], [0, "0"], [1.2]])
def test_invalid_gpu_list(tmp_path, gpus):
    with pytest.raises(ValueError, match=r"GPU|gpus"):
        replace(config(tmp_path), gpus=gpus)


def test_gpu_config_roundtrip_and_resume(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = replace(config(tmp_path), gpus=[0, "GPU-abc", "MIG-def"], concurrency=128)
    from dataclasses import asdict

    assert parse_config(json.loads(json.dumps(asdict(cfg)))) == cfg
    queued, sequential = Experiment(cfg), Experiment(replace(cfg, gpus=None))
    from pt_lm_eval.evaluations import EVALUATIONS

    step = queued.steps[0]
    evaluation = EVALUATIONS.get(step.evaluation)
    assert queued._fingerprint(step, "1", "prompt", evaluation) == sequential._fingerprint(
        sequential.steps[0], "1", "prompt", evaluation
    )


@pytest.mark.parametrize(
    "args",
    [
        {"tensor_parallel_size": 2},
        {"pipeline_parallel_size": 2},
        {"data_parallel_size": 2},
        {"port": 9000},
    ],
)
def test_reject_shared_gpu_or_port_overrides(tmp_path, args):
    cfg = config(tmp_path)
    with pytest.raises(ValueError):
        replace(cfg, server=ServerConfig(args=args))
    with pytest.raises(ValueError):
        replace(cfg, models=(replace(cfg.models[0], server_args=args),))


def test_reject_external_endpoint_and_port_overflow(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="base_url"):
        replace(cfg, models=(replace(cfg.models[0], base_url="http://localhost/v1"),))
    with pytest.raises(ValueError, match="portas"):
        replace(cfg, server=ServerConfig(port=65535))


def test_reject_output_collision_and_respect_existing_mask(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="labels únicos"):
        GPUPool(replace(cfg, models=(ModelConfig("a/b", "base"), ModelConfig("a__b", "base"))))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        GPUPool(cfg)
    assert GPUPool(replace(cfg, gpus=[2, 3])).slots == [("2", 8000), ("3", 8001)]


# A só termina quando C começar; B espera A começar. Assim C necessariamente precisa
# ocupar a vaga de B enquanto A ainda executa, sem depender de comparação de tempos.
FAKE_WORKER = r"""
import json, os, sys, time
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
cfg, gpu = payload["config"], payload["gpu"]
name = cfg["models"][0]["name"]
root = Path(cfg["output_dir"])
assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu == os.environ["PT_LM_EVAL_GPU"]
assert cfg["gpus"] is None
assert cfg["models"][0]["concurrency"] == 128
(root / (name + ".started")).write_text(json.dumps([gpu, cfg["server"]["port"]]))
wait_for = {"a": "c", "b": "a"}.get(name)
deadline = time.monotonic() + 10
while wait_for and not (root / (wait_for + ".started")).exists():
    if time.monotonic() > deadline:
        sys.exit(42)
    time.sleep(.01)
if name == "b" and os.environ["FAIL_B"] == "1":
    sys.exit(1)
result = Path(cfg["output_dir"]) / cfg["experiment"] / name / "enem.mcf.results.json"
result.parent.mkdir(parents=True, exist_ok=True)
result.write_text(json.dumps({"model": {"name": name}, "benchmark": {"name": "enem", "version": "1"},
    "evaluation": {"name": "mcf"}, "metrics": {"acc": .5}}))
"""


@pytest.mark.parametrize("fail_b", [False, True])
def test_dynamic_queue_real_subprocesses_and_consolidated_csv(tmp_path, monkeypatch, fail_b):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("FAIL_B", str(int(fail_b)))
    original_env = dict(os.environ)
    real_popen = subprocess.Popen
    launched = []

    def spawn(command, **kwargs):
        assert command[1:3] == ["-m", "pt_lm_eval.experiment.worker"]
        process = real_popen([sys.executable, "-c", FAKE_WORKER, command[-1]], **kwargs)
        launched.append((process, kwargs["stdout"]))
        return process

    monkeypatch.setattr("pt_lm_eval.experiment.pool.subprocess.Popen", spawn)
    cfg = config(tmp_path, concurrency=128)
    calls = []
    write_csv = Results.write_csv

    def consolidate(self, name):
        assert all(p.poll() is not None for p, _ in launched)
        calls.append(name)
        return write_csv(self, name)

    monkeypatch.setattr(Results, "write_csv", consolidate)
    if fail_b:
        with pytest.raises(RuntimeError, match="b"):
            Experiment(cfg).run()
    else:
        Experiment(cfg).run()
    assert calls == ["queue"]
    assert dict(os.environ) == original_env
    assert all(f.closed for _, f in launched)
    assert json.loads((tmp_path / "a.started").read_text()) == ["0", 8000]
    assert json.loads((tmp_path / "b.started").read_text()) == ["1", 8001]
    assert json.loads((tmp_path / "c.started").read_text()) == ["1", 8001]
    lines = (cfg.out_dir / "queue.csv").read_text().splitlines()
    assert len(lines) == (3 if fail_b else 4)


def test_cancel_signals_all_workers_then_waits(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    processes, files, events = [], [], []

    class Process:
        stopped = False
        interrupted = False

        def poll(self):
            if len(processes) == 2 and not self.interrupted and not self.stopped:
                self.interrupted = True
                raise KeyboardInterrupt
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True
            events.append("terminate")

        def wait(self):
            events.append("wait")

    def spawn(*args, **kwargs):
        p = Process()
        # Only the first process triggers interruption in the scheduler.
        p.interrupted = bool(processes)
        processes.append(p)
        files.append(kwargs["stdout"])
        return p

    monkeypatch.setattr("pt_lm_eval.experiment.pool.subprocess.Popen", spawn)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    with pytest.raises(KeyboardInterrupt):
        pool = GPUPool(config(tmp_path))
        pool.run(list(pool.cfg.models))
    assert events == ["terminate", "terminate", "wait", "wait"]
    assert all(f.closed for f in files)
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
    assert len(processes) == 2  # C não é iniciado depois do cancelamento.


def test_worker_sigterm_closes_endpoint_and_skips_inapplicable_steps(tmp_path, monkeypatch):
    from dataclasses import asdict

    cfg = replace(
        config(tmp_path),
        gpus=None,
        models=(ModelConfig("a", "base"),),
        benchmarks=(BenchmarkRun("enem"), BenchmarkRun("ifeval_pt")),
    )
    path = tmp_path / "worker.json"
    path.write_text(json.dumps({"gpu": "3", "config": asdict(cfg)}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("PT_LM_EVAL_GPU", "3")
    events = []

    class FakeEndpoint:
        def __enter__(self):
            events.append("start")
            return "http://localhost/v1"

        def __exit__(self, *args):
            events.append("stop")

    async def evaluate(self, steps, client):
        try:
            assert all(not s.skip_reason for s in steps)
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            await client.aclose()

    monkeypatch.setattr(Endpoint, "for_model", lambda *args: FakeEndpoint())
    monkeypatch.setattr(Experiment, "_run_model", evaluate)
    monkeypatch.setattr(Results, "write_csv", lambda *args: pytest.fail("worker escreveu CSV"))
    previous = signal.getsignal(signal.SIGTERM)
    assert worker_main(str(path)) == 130
    assert events == ["start", "stop"]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_docker_selects_single_gpu_without_host_index_inside(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("PT_LM_EVAL_GPU", "3")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="container-id")

    monkeypatch.setattr("pt_lm_eval.inference.server.subprocess.run", run)
    monkeypatch.setattr("pt_lm_eval.inference.server.subprocess.Popen", lambda *a, **kw: None)
    server = DockerVLLM("model", args={}, log_path=tmp_path / "vllm.log", image="test-image")
    with (tmp_path / "vllm.log").open("wb") as fh:
        server._start(server.serve_args(), fh)
    cmd = commands[0]
    assert cmd[cmd.index("--gpus") + 1] == "device=3"
    assert not any("VISIBLE_DEVICES=" in arg for arg in cmd)


def test_real_coordinator_cancellation_stops_servers_during_startup(tmp_path, monkeypatch):
    """CLI → workers → servidores locais reais (simulados), cancelados antes do /health."""
    import time
    from dataclasses import asdict
    from pathlib import Path

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    executable = tmp_path / "fake-vllm"
    executable.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json, os, signal, sys, time
from pathlib import Path
root = Path(__file__).parent
name = sys.argv[2]
def stop(*args):
    (root / (name + ".stopped")).touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(root / (name + ".started")).write_text(json.dumps({
    "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "pid": os.getpid(),
    "port": sys.argv[sys.argv.index("--port") + 1]}))
while True:
    time.sleep(.1)
"""
    )
    executable.chmod(0o755)
    cfg = config(tmp_path, server=ServerConfig(executable=str(executable), port=28100))
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(asdict(cfg)))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    with (tmp_path / "coordinator.log").open("wb") as fh:
        process = subprocess.Popen(
            [sys.executable, "-m", "pt_lm_eval.cli", "run", str(path)],
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 20
            while not all((tmp_path / f"{name}.started").exists() for name in ("a", "b")):
                assert process.poll() is None, (tmp_path / "coordinator.log").read_text()
                assert time.monotonic() < deadline, "workers não iniciaram os servidores"
                time.sleep(0.05)
            process.terminate()
            assert process.wait(timeout=15) != 0
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
    for name, gpu, port in (("a", "0", "28100"), ("b", "1", "28101")):
        started = json.loads((tmp_path / f"{name}.started").read_text())
        assert started["gpu"] == gpu and started["port"] == port
        assert (tmp_path / f"{name}.stopped").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(started["pid"], 0)
    assert not (tmp_path / "c.started").exists()
    assert (cfg.out_dir / "queue.csv").exists()
