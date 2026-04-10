"""
memx HTTP server

Supports multi-tenant: each endpoint accepts an optional tenant_id query parameter.
Omitting tenant_id behaves identically to single-tenant mode.
"""

import argparse
import os
import threading
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from memxcore.core import MemoryManager, MemoryResult
from memxcore.core.paths import resolve_workspace


def _default_workspace() -> str:
    fallback = os.path.join(os.path.dirname(__file__), "..")
    return resolve_workspace(fallback)


_managers: Dict[str, MemoryManager] = {}
_managers_lock = threading.Lock()


def _get_manager(tenant_id: Optional[str] = None) -> MemoryManager:
    key = tenant_id or "__default__"
    if key not in _managers:
        with _managers_lock:
            if key not in _managers:
                _managers[key] = MemoryManager(
                    workspace_path=_default_workspace(),
                    tenant_id=tenant_id,
                )
    return _managers[key]


app = FastAPI(title="memxcore")


class RememberRequest(BaseModel):
    text: str
    level: Optional[int] = None


class MemoryResultModel(BaseModel):
    content: str
    source: str
    level: int
    relevance_score: float
    metadata: dict


@app.post("/remember")
def remember(req: RememberRequest, tenant_id: Optional[str] = None) -> dict:
    path = _get_manager(tenant_id).remember(req.text, req.level)
    return {"status": "ok", "path": path}


@app.get("/search", response_model=List[MemoryResultModel])
def search(
    query: str,
    max_results: int = 10,
    tenant_id: Optional[str] = None,
) -> List[MemoryResultModel]:
    results: List[MemoryResult] = _get_manager(tenant_id).search(
        query, max_results=max_results
    )
    return [MemoryResultModel(**r.__dict__) for r in results]


@app.post("/compact")
def compact(force: bool = False, tenant_id: Optional[str] = None) -> dict:
    _get_manager(tenant_id).compact(force=force)
    return {"status": "ok"}


@app.post("/rebuild-rag")
def rebuild_rag(tenant_id: Optional[str] = None) -> dict:
    count = _get_manager(tenant_id).rebuild_rag_index()
    return {"status": "ok", "indexed": count}


def main() -> None:
    parser = argparse.ArgumentParser(description="memx server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("memxcore.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
