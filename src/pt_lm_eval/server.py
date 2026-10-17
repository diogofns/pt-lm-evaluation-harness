"""Endpoint de cada modelo: um vLLM gerenciado (processo local ou Docker) ou uma URL existente."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ModelConfig, ServerConfig

HOST = "127.0.0.1"
log = logging.getLogger("pt_lm_eval")
_DOCKER_ENV = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "HF_TOKEN", "HF_HUB_CACHE")


class ServerError(RuntimeError):
    """O servidor do modelo não subiu (porta ocupada, falha na inicialização, tempo esgotado)."""


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name.strip("/"))


@contextmanager
def model_endpoint(model: ModelConfig, server: ServerConfig, log_dir: Path) -> Iterator[str]:
    """Base URL do modelo: a dele (`base_url`) ou a de um vLLM que sobe e cai com o bloco."""
    if model.base_url:
        yield model.base_url
        return
    with vllm_server(
        model.name,
        args=server.args_for(model),
        log_path=log_dir / f"{slug(model.display_name)}.vllm.log",
        mode=server.mode,
        port=server.port,
        image=server.image,
        executable=server.executable,
        startup_timeout=server.startup_timeout,
    ) as base_url:
        yield base_url


def serve_args(model: str, port: int, args: dict[str, Any]) -> list[str]:
    """`{"max_model_len": 8192, "trust_remote_code": True}` → `--max-model-len 8192 --trust-remote-code`.

    `True` vira flag sem valor; `False` e `None` são omitidos. `--served-model-name` fixa o
    nome usado nas requisições, mesmo quando o caminho do checkpoint muda (no container).
    """
    cmd = ["serve", model, "--host", "0.0.0.0", "--port", str(port), "--served-model-name", model]
    for key, value in args.items():
        if value is not None and value is not False:
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
    """Sobe o vLLM, espera ficar pronto, confere o modelo servido e devolve a base URL."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if _port_in_use(port):  # senão o /health de outro serviço passaria pelo nosso
        raise ServerError(f"a porta {port} já está em uso; escolha outra em server.port")
    cmd = serve_args(model, port, args)
    with open(log_path, "wb") as log_file:
        if mode == "local":
            handle: Any = subprocess.Popen(
                [executable, *cmd],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            alive = lambda: handle.poll() is None
        elif mode == "docker":
            if not image:
                raise ValueError("server.image é obrigatório no modo docker")
            handle = _start_container(cmd, image, port, model, log_file)
            alive = lambda: _container_running(handle)
        else:
            raise ValueError(f"server.mode deve ser local ou docker, não {mode!r}")
        try:
            _wait_ready(port, alive, log_path, startup_timeout)
            _check_served_model(port, model)
            yield f"http://{HOST}:{port}/v1"
        finally:
            _stop(handle, mode)


def _start_container(cmd: list[str], image: str, port: int, model: str, log_file: Any) -> str:
    docker = ["docker", "run", "--rm", "-d", "--gpus", "all", "-p", f"{port}:{port}"]
    if cache := os.environ.get("HF_HUB_CACHE"):
        docker += ["-v", f"{cache}:{cache}"]
    if Path(model).exists():  # checkpoint local: monta e usa o caminho absoluto
        path = str(Path(model).resolve())
        docker += ["-v", f"{path}:{path}"]
        cmd = [cmd[0], path, *cmd[2:]]
    docker += [arg for var in _DOCKER_ENV if var in os.environ for arg in ("-e", f"{var}={os.environ[var]}")]
    docker += ["--entrypoint", "vllm", image, *cmd]
    container = subprocess.run(docker, capture_output=True, text=True, check=True).stdout.strip()
    subprocess.Popen(["docker", "logs", "-f", container], stdout=log_file, stderr=subprocess.STDOUT)
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
    start = time.monotonic()
    next_report = start + 60
    while time.monotonic() < start + timeout:
        if not alive():
            raise ServerError(f"vLLM terminou antes de ficar pronto.\n--- fim do log ---\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=5):
                log.info("vLLM pronto em %ds", time.monotonic() - start)
                return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
        if time.monotonic() >= next_report:  # download de pesos não aparece no log do vLLM
            log.info("aguardando vLLM há %ds (log: %s)", time.monotonic() - start, log_path)
            next_report += 60
    raise ServerError(f"vLLM não ficou pronto em {timeout}s.\n--- fim do log ---\n{_tail(log_path)}")


def _port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((HOST, port)) == 0


def _check_served_model(port: int, model: str) -> None:
    with urllib.request.urlopen(f"http://{HOST}:{port}/v1/models", timeout=10) as response:
        served = [m["id"] for m in json.load(response).get("data", [])]
    if model not in served:
        raise ServerError(f"o servidor na porta {port} serve {served}, não {model}")


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
        return path.read_bytes()[-max_bytes:].decode("utf-8", errors="replace")
    except OSError:
        return ""
