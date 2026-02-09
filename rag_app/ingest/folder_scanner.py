"""Scanner de pasta para aprendizagem automática de novos arquivos .docx."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rag_app.ingest.pipeline import OllamaQdrantIngestionPipeline


class FolderIngestionScanner:
    def __init__(
        self,
        watch_dir: Path,
        pipeline: OllamaQdrantIngestionPipeline,
        state_path: Path,
        poll_interval_seconds: float = 3.0,
    ) -> None:
        self._watch_dir = Path(watch_dir)
        self._pipeline = pipeline
        self._state_path = Path(state_path)
        self._poll_interval_seconds = poll_interval_seconds
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def _load_state(self) -> dict[str, float]:
        if not self._state_path.exists():
            return {}
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(key): float(value) for key, value in data.items()}

    def _save_state(self) -> None:
        self._state_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def run_once(self) -> int:
        ingested_files = 0
        for docx_path in sorted(self._watch_dir.glob("*.docx")):
            file_key = str(docx_path.resolve())
            modified_at = docx_path.stat().st_mtime
            if self._state.get(file_key) == modified_at:
                continue
            self._pipeline.ingest_docx(docx_path)
            self._state[file_key] = modified_at
            ingested_files += 1

        self._save_state()
        return ingested_files

    def run_forever(self) -> None:
        while True:
            processed = self.run_once()
            if processed:
                print(f"✅ Scanner processou {processed} arquivo(s) .docx.")
            time.sleep(self._poll_interval_seconds)
