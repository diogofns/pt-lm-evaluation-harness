"""Sobe e derruba um servidor vLLM (processo local ou container Docker)."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
_DOCKER_ENV = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "HF_TOKEN", "HF_HUB_CACHE")


def serve_args(model: str, port: int, args: dict[str, Any]) -> list[str]:
    """`{"max_model_len": 8192, "trust_remote_code": True}` → `--max-model-len 8192 --trust-remote-code`.

    `True` vira flag sem valor; `False` e `None` são omitidos.
    """
    # --served-model-name fixa o nome usado nas requisições, mesmo quando o
    # caminho do checkpoint muda (ex.: montado no container).
    cmd = ["serve", model, "--host", "0.0.0.0", "--port", str(port), "--served-model-name", model]
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        cmd += [flag] if value is True else [flag, str(value)]
    return cmd


@contextmanager
def vllm_server(
    model: str,
    *,
    args: dict[str, Any],
    log_path: Path,
    mode: str = "local",
    port: int = 8000,
    image: str | None = None,
    executable: str = "vllm",
    startup_timeout: int = 1800,
) -> Iterator[str]:
    """Context manager que devolve a base URL (`http://127.0.0.1:<port>/v1`)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = serve_args(model, port, args)
    with open(log_path, "wb") as log:
        if mode == "local":
            handle: Any = subprocess.Popen(
                [executable, *cmd],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            alive = lambda: handle.poll() is None
        elif mode == "docker":
            if not image:
                raise ValueError("server.image é obrigatório no modo docker")
            handle = _start_container(cmd, image, port, model, log)
            alive = lambda: _container_running(handle)
        else:
            raise ValueError(f"server.mode deve ser local ou docker, não {mode!r}")
        try:
            _wait_ready(port, alive, log_path, startup_timeout)
            yield f"http://{HOST}:{port}/v1"
        finally:
            _stop(handle, mode)


def _start_container(cmd: list[str], image: str, port: int, model: str, log: Any) -> str:
    docker = ["docker", "run", "--rm", "-d", "--gpus", "all", "-p", f"{port}:{port}"]
    if cache := os.environ.get("HF_HUB_CACHE"):
        docker += ["-v", f"{cache}:{cache}"]
    if Path(model).exists():  # checkpoint local: monta e usa o caminho absoluto
        path = str(Path(model).resolve())
        docker += ["-v", f"{path}:{path}"]
        cmd = [cmd[0], path, *cmd[2:]]
    for var in _DOCKER_ENV:
        if var in os.environ:
            docker += ["-e", f"{var}={os.environ[var]}"]
    docker += ["--entrypoint", "vllm", image, *cmd]
    container = subprocess.run(docker, capture_output=True, text=True, check=True).stdout.strip()
    subprocess.Popen(["docker", "logs", "-f", container], stdout=log, stderr=subprocess.STDOUT)
    return container


def _container_running(container: str) -> bool:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.returncode == 0 and out.stdout.strip() == "true"


def _wait_ready(port: int, alive: Any, log_path: Path, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive():
            raise RuntimeError(f"vLLM terminou antes de ficar pronto.\n--- fim do log ---\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=5):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise TimeoutError(f"vLLM não ficou pronto em {timeout}s.\n--- fim do log ---\n{_tail(log_path)}")


def _stop(handle: Any, mode: str, grace: int = 30) -> None:
    if mode == "docker":
        subprocess.run(["docker", "stop", "-t", str(grace), handle], capture_output=True, check=False)
        return
    if handle.poll() is not None:
        return
    os.killpg(handle.pid, signal.SIGTERM)  # o vLLM cria processos filhos
    try:
        handle.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(handle.pid, signal.SIGKILL)
        handle.wait()


def _tail(path: Path, max_bytes: int = 4000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-max_bytes:].decode("utf-8", errors="replace")
