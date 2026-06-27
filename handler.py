from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any

import runpod

from qwen_asr import Qwen3ASRModel

_HANDLER: EndpointHandler | None = None


class EndpointHandler:
    def __init__(self, path: str = "") -> None:
        self.model_id = self._resolve_model_path(path)
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            dtype="float16",
            device_map="cuda",
        )

    def _resolve_model_path(self, path: str) -> str:
        env_model_id = os.environ.get("MODEL_ID", "").strip()
        if env_model_id:
            return env_model_id

        candidate = Path(path) if path else None
        if candidate and candidate.is_dir() and (candidate / "config.json").exists():
            return str(candidate)

        return "atlasia/moulsot.v0.3"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        payload = data.get("inputs") or data.get("input") or data
        audio_path, cleanup_required = self._resolve_audio_path(payload)
        language = self._resolve_language(payload)

        try:
            result = self.model.transcribe(audio=str(audio_path), language=language)
            text, resolved_language = self._extract_text(result)

            return {
                "text": text,
                "language": resolved_language or language or "auto",
                "model": self.model_id,
            }
        finally:
            if cleanup_required:
                audio_path.unlink(missing_ok=True)

    def _resolve_language(self, payload: Any) -> str | None:
        if isinstance(payload, dict):
            value = payload.get("language") or payload.get("lang")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _resolve_audio_path(self, payload: Any) -> tuple[Path, bool]:
        if isinstance(payload, str):
            return Path(payload), False

        if isinstance(payload, (list, tuple)) and payload:
            return self._resolve_audio_path(payload[0])

        if isinstance(payload, dict):
            nested = payload.get("audio") or payload.get("file") or payload.get("input")
            if nested is not None:
                try:
                    return self._resolve_audio_path(nested)
                except ValueError:
                    pass

            path_value = payload.get("path")
            if isinstance(path_value, str) and path_value:
                return Path(path_value), False

            bytes_value = (
                payload.get("audioBase64")
                or payload.get("audio_base64")
                or payload.get("bytes")
                or payload.get("data")
            )
            if isinstance(bytes_value, str) and bytes_value:
                raw_bytes = self._decode_bytes(bytes_value)
                return self._write_temp_audio(raw_bytes), True

        raise ValueError("Unsupported audio payload. Expected a file path or audio bytes.")

    def _decode_bytes(self, value: str) -> bytes:
        if value.startswith("data:") and "," in value:
            value = value.split(",", 1)[1]
        return base64.b64decode(value)

    def _write_temp_audio(self, raw_bytes: bytes) -> Path:
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(raw_bytes)
            return Path(handle.name)

    def _extract_text(self, result: Any) -> tuple[str, str | None]:
        if isinstance(result, (list, tuple)) and result:
            return self._extract_text(result[0])

        if hasattr(result, "text"):
            text = getattr(result, "text", "")
            language = getattr(result, "language", None)
            return self._normalize_text(text), self._normalize_language(language)

        if isinstance(result, dict):
            text = result.get("text")
            language = result.get("language")
            return self._normalize_text(text), self._normalize_language(language)

        return self._normalize_text(result), None

    def _normalize_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _normalize_language(self, value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None


def _get_handler() -> EndpointHandler:
    global _HANDLER
    if _HANDLER is None:
        _HANDLER = EndpointHandler()
    return _HANDLER


def handler(job: dict[str, Any]) -> dict[str, Any]:
    return _get_handler()(job)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
