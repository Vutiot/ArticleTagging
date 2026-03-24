"""vLLM server launcher for serving fine-tuned VLMs.

Builds and launches a vLLM OpenAI-compatible API server as a subprocess.
Prefix caching is enabled by default in vLLM V1 (>= 0.8.0) — no flag needed.
"""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from article_tagging.configs.models import ServingConfig

logger = logging.getLogger(__name__)
console = Console()


def build_vllm_command(config: ServingConfig) -> list[str]:
    """Build the ``vllm serve`` CLI command from a serving configuration.

    Args:
        config: Serving configuration with model path, port, and resource limits.

    Returns:
        A list of command-line arguments suitable for :func:`subprocess.Popen`.
    """
    cmd = [
        "vllm",
        "serve",
        str(config.model_path),
        "--port",
        str(config.port),
        "--host",
        config.host,
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-model-len",
        str(config.max_model_len),
        "--dtype",
        config.dtype,
    ]

    # LoRA adapter support
    if config.adapter_path is not None:
        cmd.extend([
            "--enable-lora",
            "--lora-modules",
            f"adapter={config.adapter_path}",
        ])

    return cmd


def launch_server(config: ServingConfig) -> subprocess.Popen:
    """Launch the vLLM server as a background subprocess.

    The server exposes an OpenAI-compatible API at
    ``http://{host}:{port}/v1/chat/completions``.

    Args:
        config: Serving configuration.

    Returns:
        The :class:`subprocess.Popen` handle for the running server.
    """
    cmd = build_vllm_command(config)

    console.print(f"[bold]Launching vLLM server[/bold] on {config.host}:{config.port}")
    console.print(f"  Model: [cyan]{config.model_path}[/cyan]")
    if config.adapter_path:
        console.print(f"  LoRA adapter: [cyan]{config.adapter_path}[/cyan]")
    console.print(f"  GPU memory: {config.gpu_memory_utilization}, max tokens: {config.max_model_len}")
    console.print("  Prefix caching: [green]enabled by default[/green] (vLLM V1)")
    console.print(f"\n  Command: {' '.join(cmd)}\n")

    process = subprocess.Popen(cmd)
    logger.info("vLLM server started with PID %d", process.pid)

    return process
