"""Pydantic configuration models for the ArticleTagging pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T", bound=BaseModel)


# ─── Pagination ────────────────────────────────────────────────────────────


class PaginationType(StrEnum):
    NEXT_LINK = "next_link"
    PAGE_NUMBER = "page_number"
    INFINITE_SCROLL = "infinite_scroll"


class PaginationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: PaginationType = PaginationType.NEXT_LINK
    selector: str | None = None
    url_pattern: str | None = None
    max_pages: int = 100


# ─── SiteConfig ────────────────────────────────────────────────────────────


class SiteConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    base_url: str
    listing_selector: str
    detail_selectors: dict[str, str]
    pagination: PaginationConfig = Field(default_factory=PaginationConfig)
    use_playwright: bool = False
    rate_limit: float = 1.0
    max_listings: int | None = None
    wait_for_selector: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


# ─── DatasetConfig ─────────────────────────────────────────────────────────


class DatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_path: Path
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)
    text_only: bool = False
    system_prompt: str = (
        "You extract product attributes from the title and image. "
        "Respond with valid JSON only."
    )
    min_samples: int = 10
    deduplicate: bool = True
    category_field: str | None = None

    @field_validator("split_ratio")
    @classmethod
    def validate_split_ratio(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(v)}")
        if any(r < 0 for r in v):
            raise ValueError("Split ratios must be non-negative")
        return v


# ─── TrainingConfig ────────────────────────────────────────────────────────


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    load_in_4bit: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    epochs: int = 3
    max_steps: int = -1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    gradient_checkpointing: str = "unsloth"

    eval_steps: int = 100
    save_steps: int = 100
    early_stopping_patience: int = 3

    use_wandb: bool = False
    run_name: str | None = None

    output_dir: Path = Path("models/default-run")
    merge_on_export: bool = True


# ─── ServingConfig ─────────────────────────────────────────────────────────


class ServingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_path: Path
    adapter_path: Path | None = None
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    port: int = 8000
    host: str = "0.0.0.0"
    dtype: str = "auto"


# ─── EvalConfig ────────────────────────────────────────────────────────────


class EvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_data_path: Path
    schema_path: Path
    server_url: str = "http://localhost:8000"
    model_name: str = "default"
    output_dir: Path = Path("reports/")
    batch_concurrency: int = 8
    compare_with: list[Path] = Field(default_factory=list)


# ─── PipelineConfig ────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    site: SiteConfig | None = None
    dataset: DatasetConfig | None = None
    training: TrainingConfig | None = None
    serving: ServingConfig | None = None
    evaluation: EvalConfig | None = None


# ─── Config Loading ────────────────────────────────────────────────────────


def load_config(path: str | Path, model_cls: type[T]) -> T:
    """Load and validate a YAML config file into a Pydantic model.

    Args:
        path: Path to the YAML file.
        model_cls: The Pydantic model class to validate against.

    Returns:
        A validated instance of model_cls.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        pydantic.ValidationError: If the YAML content fails validation.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return model_cls.model_validate(raw)
