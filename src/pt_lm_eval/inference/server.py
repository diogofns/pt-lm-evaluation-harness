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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from ..experiment.config import ModelConfig, ServerConfig

HOST = "127.0.0.1"
log = logging.getLogger("pt_lm_eval")


class ServerError(RuntimeError):
    """O servidor do modelo não subiu (porta ocupada, falha na inicialização, tempo esgotado)."""


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name.strip("/"))


class Endpoint(ABC):
    @abstractmethod
    def __enter__(self) -> str: ...

    def __exit__(self, *exc: object) -> None:
        return None

    @staticmethod
    def for_model(model: ModelConfig, server: ServerConfig, log_dir: Path) -> Endpoint:
        if model.base_url:
            return ExternalEndpoint(model.base_url)
        cls = {"local": LocalVLLM, "docker": DockerVLLM}.get(server.mode)
        if cls is None:
            raise ValueError(f"server.mode deve ser local ou docker, não {server.mode!r}")
        return cls(
            model.name,
            args=server.args_for(model),
            log_path=log_dir / f"{slug(model.display_name)}.vllm.log",
            port=server.port,
            executable=server.executable,
            image=server.image,
            startup_timeout=server.startup_timeout,
        )


class ExternalEndpoint(Endpoint):
    """Um servidor que já está no ar (ou uma API)."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __enter__(self) -> str:
        return self.base_url


class VLLMServer(Endpoint):
    """Um vLLM que este processo sobe e derruba; subclasses dizem como iniciar e parar."""

    bind_host = HOST

    def __init__(
        self,
        model: str,
        *,
        args: dict[str, Any],
        log_path: Path,
        port: int = 8000,
        executable: str = "vllm",
        image: str | None = None,
        startup_timeout: int = 1800,
    ) -> None:
        self.model, self.args, self.log_path, self.port = model, args, log_path, port
        self.executable, self.image, self.startup_timeout = executable, image, startup_timeout
        self._log: BinaryIO | None = None

    @abstractmethod
    def _start(self, cmd: list[str], log: BinaryIO) -> None: ...

    @abstractmethod
    def _alive(self) -> bool: ...

    @abstractmethod
    def _stop(self, grace: int) -> None: ...

    def serve_args(self) -> list[str]:
        """`{"max_model_len": 8192, "trust_remote_code": True}` → `--max-model-len 8192 --trust-remote-code`.

        `True` vira flag sem valor; `False` e `None` são omitidos. `--served-model-name` fixa o
        nome usado nas requisições, mesmo quando o caminho do checkpoint muda (no container).
        """
        cmd = [
            "serve",
            self.model,
            "--host",
            self.bind_host,
            "--port",
            str(self.port),
            "--served-model-name",
            self.model,
        ]
        for key, value in self.args.items():
            if value is not None and value is not False:
                flag = "--" + key.replace("_", "-")
                cmd += [flag] if value is True else [flag, str(value)]
        return cmd

    def __enter__(self) -> str:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._port_in_use():
            raise ServerError(f"a porta {self.port} já está em uso; escolha outra em server.port")
        self._log = open(self.log_path, "wb")
        try:
            self._start(self.serve_args(), self._log)
            self._wait_ready()
            self._check_served_model()
        except BaseException:
            self.__exit__()
            raise
        return f"http://{HOST}:{self.port}/v1"

    def __exit__(self, *exc: object) -> None:
        try:
            self._stop(grace=30)
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None

    def _port_in_use(self) -> bool:
        with socket.socket() as sock:
            sock.settimeout(1)
            return sock.connect_ex((HOST, self.port)) == 0

    def _wait_ready(self) -> None:
        start = time.monotonic()
        next_report = start + 60
        while time.monotonic() < start + self.startup_timeout:
            if not self._alive():
                raise ServerError(f"vLLM terminou antes de ficar pronto.\n--- fim do log ---\n{self._tail()}")
            try:
                with urllib.request.urlopen(f"http://{HOST}:{self.port}/health", timeout=5):
                    log.info("vLLM pronto em %ds", time.monotonic() - start)
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(2)
            if time.monotonic() >= next_report:  # download de pesos não aparece no log do vLLM
                log.info("aguardando vLLM há %ds (log: %s)", time.monotonic() - start, self.log_path)
                next_report += 60
        raise ServerError(
            f"vLLM não ficou pronto em {self.startup_timeout}s.\n--- fim do log ---\n{self._tail()}"
        )

    def _check_served_model(self) -> None:
        with urllib.request.urlopen(f"http://{HOST}:{self.port}/v1/models", timeout=10) as response:
            served = [m["id"] for m in json.load(response).get("data", [])]
        if self.model not in served:
            raise ServerError(f"o servidor na porta {self.port} serve {served}, não {self.model}")

    def _tail(self, max_bytes: int = 4000) -> str:
        try:
            return self.log_path.read_bytes()[-max_bytes:].decode("utf-8", errors="replace")
        except OSError:
            return ""


class LocalVLLM(VLLMServer):
    """vLLM como processo local (`executable serve …`), num grupo de processos próprio."""

    _process: subprocess.Popen | None = None

    def _start(self, cmd: list[str], log: BinaryIO) -> None:
        self._process = subprocess.Popen(
            [self.executable, *cmd],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _stop(self, grace: int) -> None:
        if not self._alive():
            return
        assert self._process is not None
        os.killpg(self._process.pid, signal.SIGTERM)  # o vLLM cria processos filhos
        try:
            self._process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(self._process.pid, signal.SIGKILL)
            self._process.wait()


class DockerVLLM(VLLMServer):
    """vLLM num container (`server.image`), com o cache do HF e as GPUs visíveis repassados."""

    ENV = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "HF_TOKEN", "HF_HUB_CACHE")
    bind_host = "0.0.0.0"  # dentro do contêiner; a porta no host fica no loopback
    _container: str | None = None

    def _start(self, cmd: list[str], log: BinaryIO) -> None:
        if not self.image:
            raise ValueError("server.image é obrigatório no modo docker")
        gpu = os.environ.get("PT_LM_EVAL_GPU")
        devices = f"device={gpu}" if gpu is not None else "all"
        docker = ["docker", "run", "--rm", "-d", "--gpus", devices, "-p", f"{HOST}:{self.port}:{self.port}"]
        if cache := os.environ.get("HF_HUB_CACHE"):
            docker += ["-v", f"{cache}:{cache}"]
        if Path(self.model).exists():  # checkpoint local: monta e usa o caminho absoluto
            path = str(Path(self.model).resolve())
            docker += ["-v", f"{path}:{path}"]
            cmd = [cmd[0], path, *cmd[2:]]
        # O Docker remapeia a GPU selecionada: o índice do host não vale dentro do container.
        env_names = self.ENV if gpu is None else ("HF_TOKEN", "HF_HUB_CACHE")
        docker += [arg for var in env_names if var in os.environ for arg in ("-e", var)]
        docker += ["--entrypoint", "vllm", self.image, *cmd]
        self._container = subprocess.run(docker, capture_output=True, text=True, check=True).stdout.strip()
        subprocess.Popen(["docker", "logs", "-f", self._container], stdout=log, stderr=subprocess.STDOUT)

    def _alive(self) -> bool:
        if self._container is None:
            return False
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self._container],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.returncode == 0 and out.stdout.strip() == "true"

    def _stop(self, grace: int) -> None:
        if self._container is not None:
            subprocess.run(
                ["docker", "stop", "-t", str(grace), self._container], capture_output=True, check=False
            )
