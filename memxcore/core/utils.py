import json
import logging
import os
import re
import threading
from typing import Tuple, Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)

# ── Category validation ──────────────────────────────────────────────────────

_SAFE_CATEGORY = re.compile(r"^[a-zA-Z0-9_-]+$")

_ALLOWED_CATEGORIES = {
    "user_model", "domain", "project_state", "episodic",
    "references", "general", "permanent",
}


def sanitize_category(category: str) -> str:
    """
    Validate and sanitize a category name to prevent path traversal.
    Returns a safe category string. Falls back to 'general' for invalid input.
    """
    category = category.strip()
    if not category:
        return "general"
    if not _SAFE_CATEGORY.match(category):
        logger.warning("Rejected unsafe category: %r (path traversal attempt?)", category)
        return "general"
    if category in _ALLOWED_CATEGORIES:
        return category
    # Unknown but safe characters — allow but log
    logger.info("Non-standard category: %r", category)
    return category


# ── LLM backend ──────────────────────────────────────────────────────────────


def ensure_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8"):
            pass


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_config_key(config_path: str, key: str, value: str) -> Dict[str, Any]:
    """
    Set a dot-notation key in a YAML config file.
    Creates the file if it doesn't exist. Returns the updated config dict.
    """
    data: Dict[str, Any] = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            data = {}

    # Navigate dot notation: "llm.model" -> data["llm"]["model"]
    parts = key.split(".")
    target = data
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]

    # Auto-convert numeric/bool strings
    if value.lower() in ("true", "false"):
        target[parts[-1]] = value.lower() == "true"
    elif value.isdigit():
        target[parts[-1]] = int(value)
    else:
        try:
            target[parts[-1]] = float(value)
        except ValueError:
            target[parts[-1]] = value

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)
    return data


def load_config(workspace_path: str, pkg_dir: str) -> Dict[str, Any]:
    """
    Load config from workspace config.yaml, falling back to package bundled one.
    Returns the merged config dict.
    """
    from .paths import resolve_install_dir
    root_dir = resolve_install_dir(workspace_path)
    ws_config = os.path.join(root_dir, "config.yaml")
    if os.path.isfile(ws_config):
        try:
            with open(ws_config, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            pass
    pkg_config = os.path.join(pkg_dir, "config.yaml")
    if os.path.isfile(pkg_config):
        try:
            with open(pkg_config, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            pass
    return {}


def parse_front_matter(raw: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse YAML front matter from a Markdown file.
    If no front matter is present, returns an empty dict and the original text.
    """
    raw = raw.lstrip()
    if not raw.startswith("---"):
        return {}, raw

    parts = raw.split("\n", 1)
    if len(parts) < 2:
        return {}, raw

    rest = parts[1]
    end_idx = rest.find("\n---")
    if end_idx == -1:
        return {}, raw

    yaml_block = rest[:end_idx]
    body = rest[end_idx + len("\n---") :].lstrip("\n")

    try:
        meta = yaml.safe_load(yaml_block) or {}
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}
    return meta, body


def extract_simple_summary(body: str, max_len: int = 160) -> str:
    """
    Minimal summary: take the first non-empty line, truncated to max_len.
    """
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > max_len:
            return line[: max_len - 3] + "..."
        return line
    return ""


def _detect_default_model() -> str:
    """Pick a default model based on which API key is available."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-haiku-4-5-20251001"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-4o-mini"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-2.0-flash"
    if os.environ.get("OLLAMA_API_BASE"):
        return "ollama/llama3"
    return "anthropic/claude-haiku-4-5-20251001"


def call_llm(prompt: str, config: Dict[str, Any], max_tokens: int = 1024) -> str:
    """
    Shared LLM call utility via litellm (supports Anthropic, OpenAI, Gemini, Ollama, etc.).
    Returns an empty string when the call fails; no exceptions raised.
    Errors are logged for diagnostics.

    Config example (config.yaml):
      llm:
        model: openai/gpt-4o              # litellm format: provider/model
        api_key_env: OPENAI_API_KEY        # env var name to read API key from
    """
    llm_cfg = config.get("llm", {})
    model = llm_cfg.get("model") or _detect_default_model()

    # Resolve API key from configured env var
    api_key_env = llm_cfg.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    # If no explicit key, check common provider env vars that litellm reads automatically
    # Note: ANTHROPIC_AUTH_TOKEN is a Claude Code session token, NOT a litellm API key
    if not api_key:
        has_any_key = any(
            os.environ.get(k)
            for k in ("ANTHROPIC_API_KEY",
                       "OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_API_BASE")
        )
        if not has_any_key:
            logger.debug("call_llm: no LLM API key found in environment, skipping")
            return ""

    try:
        from litellm import completion
    except ImportError:
        logger.warning("call_llm: litellm not installed. Run: pip install litellm")
        return ""

    try:
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if api_key:
            kwargs["api_key"] = api_key

        # Pass through optional base_url for gateway/proxy setups
        base_url = llm_cfg.get("base_url") or os.environ.get("LLM_BASE_URL")
        if base_url:
            kwargs["api_base"] = base_url

        response = completion(**kwargs)
    except Exception as e:
        logger.warning("call_llm: %s", e)
        return ""

    try:
        text = response.choices[0].message.content or ""
    except (IndexError, AttributeError):
        logger.warning("call_llm: unexpected response structure")
        return ""

    # Check truncation
    finish = getattr(response.choices[0], "finish_reason", None)
    if finish == "length":
        logger.warning(
            "call_llm: response truncated at max_tokens=%d (model=%s). "
            "Consider increasing max_tokens.",
            max_tokens, model,
        )

    return text


def parse_llm_json(raw: str) -> Any:
    """Parse JSON returned by the LLM, tolerant of markdown code block wrappers."""
    raw = re.sub(r'^```[a-z]*\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return None


