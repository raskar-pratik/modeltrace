"""FastAPI application and REST endpoints for ModelTrace.

Enforces:
- Inbound Host and Origin validation against explicit localhost hosts/ports (`validate_inbound_localhost_request` — never trusts an arbitrary Host header to define same-origin)
- Request payload size cap (`MODELTRACE_MAX_UPLOAD_BYTES`)
- JSON Content-Type check on state-changing requests with bodies
- Full secret redaction in error responses and payloads
- Non-blocking background worker execution with progress polling
- Single-origin static asset and SPA serving for `frontend/dist`
"""

import os
import platform
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import fastapi
import httpx
import numpy as np
import pydantic
import scipy
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from modeltrace import APP_NAME, APP_TAGLINE, __version__
from modeltrace.adapters.local_transformers import detect_local_inference_dependencies
from modeltrace.config import ROOT_DIR, get_env_settings, utc_now_iso
from modeltrace.db.repository import Repository
from modeltrace.schemas import (
    AuditCreateRequest,
    AuditPreviewRequest,
    CalibrationImportRequest,
    DemoCalibrationCreateRequest,
    ProbeImportRequest,
    ProbeRevisionCreate,
    ProfileCreate,
    ProfileUpdate,
    SettingsUpdateRequest,
)
from modeltrace.services.exporter import ExportService
from modeltrace.services.probes import parse_probe_import_payload
from modeltrace.services.worker import AuditWorkerService
from modeltrace.utils.security import (
    DEFAULT_ALLOWED_LOCALHOST_HOSTS,
    SecurityViolationError,
    redact_secrets,
    redact_text,
    validate_inbound_localhost_request,
)


def create_app(
    db_path: Optional[str] = None,
    exports_dir: Optional[str] = None,
    start_worker: bool = True,
) -> FastAPI:
    """Create and configure a ModelTrace FastAPI application instance."""
    env = get_env_settings()
    resolved_db = db_path or env.db_path
    resolved_exports = exports_dir or env.exports_dir

    repo = Repository(db_path=resolved_db, auto_migrate=True)
    worker = AuditWorkerService(repo=repo, start_background_loop=start_worker)
    exporter = ExportService(repo=repo, exports_dir=resolved_exports)

    app = FastAPI(
        title=APP_NAME,
        description=APP_TAGLINE,
        version=__version__,
    )
    app.state.repo = repo
    app.state.worker = worker
    app.state.exporter = exporter
    app.state.env_settings = env

    app.add_middleware(
        CORSMiddleware,
        allow_origins=env.allowed_frontend_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "Origin"],
    )

    @app.middleware("http")
    async def security_host_origin_and_json_guard(request: Request, call_next: Any) -> Response:
        method = request.method.upper()
        host_header = request.headers.get("host")
        origin_header = request.headers.get("origin")
        referer_header = request.headers.get("referer")

        try:
            validate_inbound_localhost_request(
                host_header=host_header,
                origin_header=origin_header,
                referer_header=referer_header,
                method=method,
                allowed_frontend_origins=env.allowed_frontend_origins,
                allowed_localhost_hosts=DEFAULT_ALLOWED_LOCALHOST_HOSTS,
            )
        except SecurityViolationError as exc:
            status = 400 if exc.code in ("MISSING_HOST_HEADER", "INVALID_HOST_HEADER", "DISALLOWED_HOST_HEADER") else 403
            return JSONResponse(
                status_code=status,
                content={
                    "error_code": exc.code,
                    "detail": redact_text(exc.message),
                },
            )

        if method in ("POST", "PUT", "PATCH"):
            content_len = int(request.headers.get("content-length") or "0")
            max_upload = int(getattr(env, "max_upload_bytes", 2_097_152))
            if content_len > max_upload:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error_code": "UPLOAD_TOO_LARGE",
                        "detail": f"Request payload ({content_len} bytes) exceeds maximum allowed ({max_upload} bytes).",
                    },
                )
            if content_len > 0:
                ctype = (request.headers.get("content-type") or "").lower()
                if not ctype.startswith("application/json"):
                    return JSONResponse(
                        status_code=415,
                        content={
                            "error_code": "JSON_CONTENT_TYPE_REQUIRED",
                            "detail": "State-changing requests with a body must use Content-Type: application/json.",
                        },
                    )
        return await call_next(request)

    @app.exception_handler(SecurityViolationError)
    async def handle_security_violation(
        request: Request, exc: SecurityViolationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error_code": exc.code,
                "detail": redact_text(exc.message),
            },
        )

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error_code": "VALIDATION_ERROR",
                "detail": redact_text(str(exc)),
            },
        )

    @app.on_event("shutdown")
    def _shutdown_worker() -> None:
        worker.stop_worker_thread()

    # ------------------------------------------------------------------
    # Health & Runtime Capabilities
    # ------------------------------------------------------------------
    @app.get("/api/health")
    def get_health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "app_name": APP_NAME,
            "tagline": APP_TAGLINE,
            "version": __version__,
            "timestamp_utc": utc_now_iso(),
        }

    @app.get("/api/capabilities")
    def get_capabilities() -> Dict[str, Any]:
        hf_ok, hf_missing, hf_versions = detect_local_inference_dependencies()
        settings = repo.get_settings()
        provider_env_keys = sorted(
            k for k in os.environ.keys() if k.startswith("MODELTRACE_PROVIDER_")
        )
        return {
            "app_name": APP_NAME,
            "tagline": APP_TAGLINE,
            "version": __version__,
            "runtime": {
                "python_version": platform.python_version(),
                "sqlite_version": sqlite3.sqlite_version,
                "fastapi_version": fastapi.__version__,
                "pydantic_version": pydantic.__version__,
                "httpx_version": httpx.__version__,
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "local_models_dir": "./data/local_models",
            },
            "adapters": {
                "synthetic_demo": {
                    "available": True,
                    "supports_local_logit_scoring": True,
                    "is_demo": True,
                },
                "openai_compatible": {
                    "available": True,
                    "supports_local_logit_scoring": False,
                    "is_demo": False,
                    "socket_ip_pinning": True,
                    "trust_env_proxies": False,
                },
                "local_transformers": {
                    "available": hf_ok,
                    "missing_dependencies": hf_missing,
                    "installed_versions": hf_versions,
                    "supports_offline_sharded_checkpoints": True,
                    "trust_remote_code": False,
                    "local_files_only": True,
                },
            },
            "security": {
                "secret_env_prefix": "MODELTRACE_PROVIDER_",
                "configured_provider_env_names": provider_env_keys,
                "allowed_outbound_origins": settings["allowed_outbound_origins"],
                "allowed_local_inference_origins": settings["allowed_local_inference_origins"],
                "allowed_frontend_origins": env.allowed_frontend_origins,
                "allowed_localhost_hosts": sorted(DEFAULT_ALLOWED_LOCALHOST_HOSTS),
                "redirects_disabled": True,
                "tls_verification_enforced": True,
                "socket_ip_pinning_enforced": True,
            },
        }

    # ------------------------------------------------------------------
    # Profiles CRUD & Connection Test
    # ------------------------------------------------------------------
    @app.get("/api/profiles")
    def list_profiles() -> Dict[str, Any]:
        return {"profiles": redact_secrets(repo.list_profiles())}

    @app.post("/api/profiles", status_code=201)
    def create_profile(req: ProfileCreate) -> Dict[str, Any]:
        prof = repo.create_profile(req)
        return redact_secrets(prof)

    @app.get("/api/profiles/{profile_id}")
    def get_profile(profile_id: str) -> Dict[str, Any]:
        prof = repo.get_profile(profile_id)
        if not prof:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
        return redact_secrets(prof)

    @app.put("/api/profiles/{profile_id}")
    def update_profile(profile_id: str, req: ProfileUpdate) -> Dict[str, Any]:
        prof = repo.update_profile(profile_id, req)
        if not prof:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
        return redact_secrets(prof)

    @app.delete("/api/profiles/{profile_id}")
    def delete_profile(profile_id: str) -> Dict[str, Any]:
        deleted = repo.delete_profile(profile_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
        return {"deleted": True, "profile_id": profile_id}

    @app.post("/api/profiles/{profile_id}/test-connection")
    def test_profile_connection(
        profile_id: str,
        execute_network_probe: bool = Query(default=True),
    ) -> Dict[str, Any]:
        prof = repo.get_profile(profile_id)
        if not prof:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
        cap = worker.test_profile_connection(
            profile_id, execute_network_probe=execute_network_probe
        )
        return redact_secrets(cap)

    # ------------------------------------------------------------------
    # Probe Browse / Import / Edit into Immutable Revisions
    # ------------------------------------------------------------------
    @app.get("/api/probes")
    def get_default_probes() -> Dict[str, Any]:
        return repo.get_default_probe_revision()

    @app.get("/api/probes/revisions")
    def list_probe_revisions(include_prompts: bool = Query(default=False)) -> Dict[str, Any]:
        return {"revisions": repo.list_probe_revisions(include_prompts=include_prompts)}

    @app.get("/api/probes/revisions/{revision_id}")
    def get_probe_revision(revision_id: str) -> Dict[str, Any]:
        rev = repo.get_probe_revision(revision_id)
        if not rev:
            raise HTTPException(
                status_code=404, detail=f"Probe revision '{revision_id}' not found."
            )
        return rev

    @app.post("/api/probes/revisions", status_code=201)
    def create_probe_revision(req: ProbeRevisionCreate) -> Dict[str, Any]:
        return repo.create_probe_revision(req)

    @app.post("/api/probes/import", status_code=201)
    def import_probes(req: ProbeImportRequest) -> Dict[str, Any]:
        items = parse_probe_import_payload(req.format, req.raw_content)
        rev_req = ProbeRevisionCreate(
            name=req.name,
            description=req.description,
            parent_revision_id=None,
            prompts=items,
        )
        return repo.create_probe_revision(rev_req)

    # ------------------------------------------------------------------
    # Calibrations
    # ------------------------------------------------------------------
    @app.get("/api/calibrations")
    def list_calibrations() -> Dict[str, Any]:
        return {"calibrations": repo.list_calibrations()}

    @app.get("/api/calibrations/{calibration_id}")
    def get_calibration(calibration_id: str) -> Dict[str, Any]:
        cal = repo.get_calibration(calibration_id)
        if not cal:
            raise HTTPException(
                status_code=404, detail=f"Calibration '{calibration_id}' not found."
            )
        return cal

    @app.post("/api/calibrations/import", status_code=201)
    def import_calibration(req: CalibrationImportRequest) -> Dict[str, Any]:
        return repo.import_calibration(req)

    @app.post("/api/calibrations/run-demo", status_code=201)
    def run_demo_calibration(req: DemoCalibrationCreateRequest) -> Dict[str, Any]:
        return worker.run_demo_calibration(req)

    # ------------------------------------------------------------------
    # Audits Preview / Create / Start / History / Detail / Progress / Cancel / Resume / Export
    # ------------------------------------------------------------------
    @app.post("/api/audits/preview")
    def preview_audit(req: AuditPreviewRequest) -> Dict[str, Any]:
        return redact_secrets(worker.preview_audit(req))

    @app.post("/api/audits", status_code=201)
    def create_audit(req: AuditCreateRequest) -> Dict[str, Any]:
        return redact_secrets(worker.create_audit(req))

    @app.post("/api/audits/{audit_id}/start")
    def start_audit(
        audit_id: str,
        synchronous: bool = Query(default=False),
    ) -> Dict[str, Any]:
        try:
            audit = worker.start_audit(audit_id, run_synchronously=synchronous)
            return redact_secrets(audit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/audits")
    def list_audits() -> Dict[str, Any]:
        return {"audits": redact_secrets(repo.list_audits())}

    @app.get("/api/audits/{audit_id}")
    def get_audit_detail(audit_id: str) -> Dict[str, Any]:
        audit = repo.get_audit(audit_id)
        if not audit:
            raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")
        prompt_scores = repo.list_prompt_scores(audit_id)
        attempts = repo.list_audit_attempts(audit_id)
        responses = repo.list_audit_responses(audit_id)
        exports = repo.list_exports(audit_id)
        return redact_secrets(
            {
                "audit": audit,
                "prompt_scores": prompt_scores,
                "attempts": attempts,
                "responses": responses,
                "exports": exports,
            }
        )

    @app.get("/api/audits/{audit_id}/responses")
    def get_audit_responses(audit_id: str) -> Dict[str, Any]:
        audit = repo.get_audit(audit_id)
        if not audit:
            raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")
        responses = repo.list_audit_responses(audit_id)
        return redact_secrets(
            {
                "audit_id": audit_id,
                "response_count": len(responses),
                "responses": responses,
            }
        )

    @app.post("/api/audits/{audit_id}/purge-evidence")
    def purge_audit_evidence(audit_id: str) -> Dict[str, Any]:
        try:
            return repo.purge_audit_evidence(audit_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/audits/{audit_id}/progress")
    def get_audit_progress(audit_id: str) -> Dict[str, Any]:
        try:
            return redact_secrets(worker.get_audit_progress(audit_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/audits/{audit_id}/events")
    def get_audit_events(audit_id: str) -> Dict[str, Any]:
        audit = repo.get_audit(audit_id)
        if not audit:
            raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")
        return {"audit_id": audit_id, "events": redact_secrets(repo.list_audit_events(audit_id))}

    @app.post("/api/audits/{audit_id}/cancel")
    def cancel_audit(audit_id: str) -> Dict[str, Any]:
        try:
            return redact_secrets(worker.cancel_audit(audit_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/audits/{audit_id}/resume")
    def resume_audit(
        audit_id: str,
        synchronous: bool = Query(default=False),
    ) -> Dict[str, Any]:
        try:
            return redact_secrets(
                worker.resume_audit(audit_id, run_synchronously=synchronous)
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/api/audits/{audit_id}")
    def delete_audit(audit_id: str) -> Dict[str, Any]:
        deleted = repo.delete_audit(audit_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found.")
        return {"deleted": True, "audit_id": audit_id}

    @app.get("/api/audits/{audit_id}/export")
    def export_audit(
        audit_id: str,
        format: str = Query(default="json", regex="^(json|csv|html)$"),
    ) -> Response:
        try:
            content, media_type, exp_rec = exporter.export_audit(audit_id, fmt=format)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        headers = {
            "X-ModelTrace-Export-Id": exp_rec["id"],
            "X-ModelTrace-Integrity-SHA256": exp_rec["content_hash"],
            "Content-Disposition": f'attachment; filename="{audit_id}_report.{format}"',
        }
        return PlainTextResponse(content=content, media_type=media_type, headers=headers)

    # ------------------------------------------------------------------
    # Settings & Retention
    # ------------------------------------------------------------------
    @app.get("/api/settings")
    def get_settings() -> Dict[str, Any]:
        return repo.get_settings()

    @app.put("/api/settings")
    def update_settings(req: SettingsUpdateRequest) -> Dict[str, Any]:
        return repo.update_settings(req)

    @app.post("/api/settings/retention/run")
    def run_retention(
        older_than_days: Optional[int] = Query(default=None, ge=1),
        preserve_manifests: bool = Query(default=True),
    ) -> Dict[str, Any]:
        return repo.run_retention_cleanup(
            older_than_days=older_than_days,
            preserve_manifests=preserve_manifests,
        )

    # ------------------------------------------------------------------
    # Single-Origin Frontend Serving (`frontend/dist`)
    # ------------------------------------------------------------------
    dist_dir = ROOT_DIR / "frontend" / "dist"

    @app.get("/favicon.svg", include_in_schema=False)
    @app.get("/logo.svg", include_in_schema=False)
    @app.get("/modeltrace-mark.svg", include_in_schema=False)
    def serve_favicon(request: Request) -> Response:
        asset_name = request.url.path.rsplit("/", 1)[-1]
        fav_path = dist_dir / asset_name
        if not fav_path.is_file():
            fav_path = ROOT_DIR / "frontend" / "public" / asset_name
        if fav_path.is_file():
            return Response(
                content=fav_path.read_text(encoding="utf-8"),
                media_type="image/svg+xml",
            )
        raise HTTPException(status_code=404, detail="Brand asset not found")

    @app.get("/assets/{asset_name}", include_in_schema=False)
    def serve_asset(asset_name: str) -> Response:
        if ".." in asset_name or "/" in asset_name or "\\" in asset_name:
            raise HTTPException(status_code=400, detail="Invalid asset name")
        asset_path = dist_dir / "assets" / asset_name
        if not asset_path.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")
        if asset_name.endswith(".js"):
            mtype = "application/javascript; charset=utf-8"
        elif asset_name.endswith(".css"):
            mtype = "text/css; charset=utf-8"
        elif asset_name.endswith(".svg"):
            mtype = "image/svg+xml"
        else:
            mtype = "application/octet-stream"
        return Response(content=asset_path.read_bytes(), media_type=mtype)

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> Response:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"API route '/{full_path}' not found.")
        index_path = dist_dir / "index.html"
        if index_path.is_file():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
        return HTMLResponse(
            content="<h1>ModelTrace UI build not found. Run npm --prefix frontend run build</h1>",
            status_code=200,
        )

    return app


app = create_app()
