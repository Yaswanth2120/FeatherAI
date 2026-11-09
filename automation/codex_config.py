"""Codex model selection and deterministic classification of non-retryable CLI failures.

Resolution order (first match wins):
1. ``FEATHER_CODEX_MODEL`` environment variable.
2. ``codex.model`` in ``automation/config.yaml`` (committed, reviewed).
3. A model recorded by ``feather automation doctor --probe --select`` in
   ``.feather/automation/codex-model.json`` (runtime evidence, not committed).
4. No ``-m`` flag: the Codex CLI uses its own configured default.

Only the model slug and failure categories are read or recorded; credentials are never touched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

import yaml

MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}")
ENV_MODEL = "FEATHER_CODEX_MODEL"

# Categories that will not succeed on retry without a human action.
NON_RETRYABLE = (
    ("model_entitlement", re.compile(
        r"model_not_found|not supported when using Codex with a ChatGPT account|"
        r"does not have access to (?:the )?model|model .{0,40} does not exist", re.I)),
    ("cli_outdated", re.compile(r"requires a newer version of Codex|please upgrade to the latest", re.I)),
    ("authentication", re.compile(
        r"\b401\b|unauthorized|not logged in|please (?:log ?in|sign in)|authentication (?:failed|required)|"
        r"invalid api key|token (?:has )?expired", re.I)),
    ("usage_limit", re.compile(r"usage limit|insufficient_quota|you've hit your", re.I)),
)
ACTIONS = {
    "model_entitlement": "The selected model is not available to this Codex account. Choose another model with "
                         "FEATHER_CODEX_MODEL or `feather automation doctor --probe --select`.",
    "cli_outdated": "The installed Codex CLI is too old for the selected model. Upgrade it yourself (for example "
                    "`codex update`) and rerun `feather automation doctor --probe`.",
    "authentication": "Codex is not authenticated. Run `codex login` yourself; Feather never handles credentials.",
    "usage_limit": "The Codex account reported a usage limit. Wait for the limit to reset; do not retry in a loop.",
}


class ConfigError(ValueError):
    pass


def validate_model(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not MODEL_PATTERN.fullmatch(value):
        raise ConfigError(f"Invalid Codex model identifier: {value!r}")
    return value


def load_config(root: Path) -> dict:
    path = root / "automation" / "config.yaml"
    data: dict = {}
    if path.is_file():
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as error:
            raise ConfigError(f"Invalid automation config: {error}") from error
    if not isinstance(data, dict) or data.get("version", 1) != 1:
        raise ConfigError("automation/config.yaml must be a mapping with version: 1")
    codex = data.get("codex") or {}
    if not isinstance(codex, dict) or set(codex) - {"model"}:
        raise ConfigError("codex section supports only: model")
    timeout = data.get("task_timeout_seconds", 1800)
    attempts = data.get("max_attempts", 3)
    if type(timeout) is not int or not 60 <= timeout <= 4 * 3600:
        raise ConfigError("task_timeout_seconds must be an integer from 60 to 14400")
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ConfigError("max_attempts must be an integer from 1 to 3")
    return {"model": validate_model(codex.get("model")), "task_timeout_seconds": timeout, "max_attempts": attempts}


def selection_path(root: Path) -> Path:
    return root / ".feather" / "automation" / "codex-model.json"


def resolve_model(root: Path, environ: dict | None = None) -> dict:
    environ = os.environ if environ is None else environ
    if environ.get(ENV_MODEL):
        return {"model": validate_model(environ[ENV_MODEL]), "source": f"environment {ENV_MODEL}"}
    configured = load_config(root)["model"]
    if configured:
        return {"model": configured, "source": "automation/config.yaml"}
    path = selection_path(root)
    if path.is_file():
        try:
            record = json.loads(path.read_text())
            if record.get("status") == "accepted":
                return {"model": validate_model(record.get("model")), "source": str(path.relative_to(root)),
                        "probed_at": record.get("probed_at"), "cli_version": record.get("cli_version")}
        except (OSError, ValueError):
            pass
    return {"model": None, "source": "codex CLI default (no -m flag)"}


def _messages(text: str) -> list[str]:
    """Return error messages from Codex JSONL events and plain stderr-style lines only.

    Agent prose is ignored so that a model *discussing* authentication cannot block a task.
    """
    messages = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            if re.match(r"^(error|fatal)\b", stripped, re.I) or " ERROR " in stripped:
                messages.append(stripped)
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error" and isinstance(event.get("message"), str):
            messages.append(event["message"])
        elif event.get("type") == "turn.failed":
            error = event.get("error")
            messages.append(error.get("message", "") if isinstance(error, dict) else str(error))
    return messages


def classify_failure(text: str) -> dict | None:
    for message in _messages(text):
        if "Model metadata for" in message and "Defaulting to fallback" in message:
            continue  # Client-side warning; the request may still succeed.
        for kind, pattern in NON_RETRYABLE:
            if pattern.search(message):
                return {"kind": kind, "message": _redact(message)[:500], "action": ACTIONS[kind]}
    return None


def _redact(message: str) -> str:
    return re.sub(r"(?i)(bearer\s+|sk-|token[\"':= ]+)[A-Za-z0-9._-]{8,}", r"\1[redacted]", message)
