"""Guided JSON schema generator for vLLM's guided_json parameter.

Converts dataset schema YAML definitions into JSON Schema objects that constrain
vLLM's output to valid attribute combinations, ensuring 99%+ valid JSON output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class AttributeDefinition(BaseModel):
    """A single attribute in a dataset schema."""

    name: str
    type: Literal["enum", "string", "number"]
    values: list[str] | None = None
    required: bool = True
    depends_on: str | None = None

    @model_validator(mode="after")
    def validate_enum_has_values(self) -> AttributeDefinition:
        if self.type == "enum" and not self.values:
            msg = f"Attribute '{self.name}' is type 'enum' but has no values defined"
            raise ValueError(msg)
        return self


class DatasetSchema(BaseModel):
    """Top-level dataset schema loaded from YAML."""

    name: str
    category_field: str | None = None
    attributes: list[AttributeDefinition]


def load_schema(path: Path) -> DatasetSchema:
    """Load a dataset schema from a YAML file.

    Args:
        path: Path to the YAML schema file.

    Returns:
        Parsed and validated DatasetSchema.

    Raises:
        FileNotFoundError: If the schema file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
        pydantic.ValidationError: If the YAML content does not match the schema.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetSchema.model_validate(raw)


def generate_json_schema(schema: DatasetSchema) -> dict:
    """Generate a JSON Schema dict from a DatasetSchema.

    The returned schema is suitable for passing to vLLM's ``guided_json``
    parameter to constrain model output.

    Args:
        schema: The dataset schema defining all attributes.

    Returns:
        A JSON Schema dict with ``type``, ``properties``, ``required``,
        and ``additionalProperties`` keys.
    """
    properties: dict[str, dict] = {}
    required: list[str] = []

    for attr in schema.attributes:
        if attr.type == "enum":
            properties[attr.name] = {"type": "string", "enum": attr.values}
        elif attr.type == "string":
            properties[attr.name] = {"type": "string"}
        elif attr.type == "number":
            properties[attr.name] = {"type": "number"}

        if attr.required:
            required.append(attr.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def generate_json_schema_for_category(
    schema: DatasetSchema,
    category: str,  # noqa: ARG001
) -> dict:
    """Generate a JSON Schema filtered for a specific category value.

    This is intended for future use where attributes that ``depends_on`` a
    specific category value would be filtered. For now, it returns the full
    schema (all attributes).

    Args:
        schema: The dataset schema defining all attributes.
        category: The category value to filter by (currently unused).

    Returns:
        A JSON Schema dict (same format as :func:`generate_json_schema`).
    """
    return generate_json_schema(schema)
