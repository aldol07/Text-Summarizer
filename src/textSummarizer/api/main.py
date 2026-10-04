"""FastAPI application.

Run locally:  uvicorn textSummarizer.api.main:app --reload
Then open http://127.0.0.1:8000 (UI) or http://127.0.0.1:8000/docs (OpenAPI).
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from textSummarizer import __version__
from textSummarizer.api.engine import SummarizerEngine
from textSummarizer.api.ingest import IngestError, extract_file, fetch_url
from textSummarizer.api.schemas import CompareRequest, SummarizeRequest, UrlRequest
from textSummarizer.api.settings import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = SummarizerEngine(settings)
        yield

    app = FastAPI(
        title="Text Summarizer API",
        version=__version__,
        description="Classical NLP extractive summarizers + an LSTM pointer-generator (ONNX).",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def engine(request: Request) -> SummarizerEngine:
        return request.app.state.engine

    # Handlers are sync `def`: FastAPI runs them in a worker thread, so CPU-bound
    # summarization does not block the event loop.
    @app.get("/health", tags=["meta"])
    def health(request: Request):
        eng = engine(request)
        return {"status": "ok", "version": __version__, "abstractive": eng.abstractive_available}

    @app.get("/api/v1/methods", tags=["summarize"])
    def methods(request: Request):
        return engine(request).methods()

    @app.post("/api/v1/summarize", tags=["summarize"])
    def summarize(body: SummarizeRequest, request: Request):
        try:
            return engine(request).summarize(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/compare", tags=["summarize"])
    def compare(body: CompareRequest, request: Request):
        try:
            return engine(request).compare(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/extract/url", tags=["ingest"])
    def extract_url(body: UrlRequest):
        try:
            return fetch_url(body.url).to_dict(settings.max_text_chars)
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/extract/file", tags=["ingest"])
    def extract_upload(file: Annotated[UploadFile, File()]):
        content = file.file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413, detail=f"File too large (limit {settings.max_upload_bytes // 2**20} MB)"
            )
        try:
            return extract_file(file.filename or "", content).to_dict(settings.max_text_chars)
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Mounted last so /api and /health take precedence. In production the UI lives on Vercel.
    if settings.serve_frontend and settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


app = create_app()
