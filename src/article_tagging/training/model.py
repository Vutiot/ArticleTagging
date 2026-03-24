"""Model loading utilities for VLM fine-tuning with Unsloth."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from article_tagging.configs.models import TrainingConfig

_UNSLOTH_INSTALL_MSG = (
    "Unsloth is required for model loading but is not installed.\n"
    "Install the training extras with:  pip install article-tagging[training]"
)

_TRANSFORMERS_INSTALL_MSG = (
    "Transformers is required for tokenizer loading but is not installed.\n"
    "Install the training extras with:  pip install article-tagging[training]"
)


def load_model(config: TrainingConfig) -> tuple[Any, Any]:
    """Load a vision-language model with 4-bit quantization and apply LoRA adapters.

    Uses Unsloth's ``FastVisionModel`` for memory-efficient loading (targeting
    RTX 4060 8 GB: ~2 GB weights + ~1 GB LoRA adapters in 4-bit).

    Args:
        config: Training configuration containing model name, quantization,
            and LoRA hyper-parameters.

    Returns:
        A ``(model, tokenizer)`` tuple ready for fine-tuning.

    Raises:
        ImportError: If *unsloth* is not installed.
    """
    try:
        from unsloth import FastVisionModel
    except ImportError:
        raise ImportError(_UNSLOTH_INSTALL_MSG) from None

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config.model_name,
        load_in_4bit=config.load_in_4bit,
        max_seq_length=2048,
        dtype=None,
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        use_gradient_checkpointing=config.gradient_checkpointing,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        lora_dropout=0,
        bias="none",
    )

    return model, tokenizer


def load_tokenizer(config: TrainingConfig) -> Any:
    """Load only the processor/tokenizer (no GPU required).

    This is useful for dataset preparation and pre-processing steps that
    do not need the full model weights.

    Args:
        config: Training configuration (only ``model_name`` is used).

    Returns:
        A ``transformers.AutoProcessor`` instance for the configured model.

    Raises:
        ImportError: If *transformers* is not installed.
    """
    try:
        from transformers import AutoProcessor
    except ImportError:
        raise ImportError(_TRANSFORMERS_INSTALL_MSG) from None

    return AutoProcessor.from_pretrained(config.model_name)
