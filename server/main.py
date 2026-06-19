from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import aiofiles
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .pipeline import Pipeline, TaskProgress
from .templates import (
    MeetingType,
    TemplateCreateRequest,
    TemplateManager,
    TemplateRecord,
    TemplateUpdateRequest,
)


Base = declarative_base()

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./meetings.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
CHUNK_DIR = os.environ.get("CHUNK_DIR", "./chunks")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class MeetingORM(Base):
    __tablename__ = "meetings"

    id = Column(String, primary_key=True)
    title = Column(String(500), default="")
    meeting_type = Column(String(50), default="client_consultation")
    case_id = Column(String(200), nullable=True)
    case_match_status = Column(String(50), default="pending")
    participants = Column(Text, default="[]")
    audio_filename = Column(String(500), default="")
    duration_seconds = Column(Float, default=0.0)
    transcript = Column(Text, default="")
    segments = Column(Text, default="[]")
    summary = Column(Text, default="{}")
    state = Column(String(50), default="uploading")
    error_message = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)


Base.metadata.create_all(bind=engine)

template_manager = TemplateManager()

task_progress_store: dict[str, TaskProgress] = {}
task_ws_connections: dict[str, list[WebSocket]] = {}
task_queue: asyncio.Queue[str] = asyncio.Queue()

pipeline = Pipeline(
    template_manager=template_manager,
    whisper_model_size=os.environ.get("WHISPER_MODEL_SIZE", "medium"),
    openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
    openai_base_url=os.environ.get("OPENAI_BASE_URL"),
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    skip_diarization=os.environ.get("SKIP_DIARIZATION", "false").lower() == "true",
)


class MeetingCreateRequest(BaseModel):
    title: str = Field("", max_length=500)
    meeting_type: MeetingType = MeetingType.CLIENT_CONSULTATION
    participants: list[str] = Field(default_factory=list)
    case_background: Optional[str] = None


class MeetingResponse(BaseModel):
    id: str
    title: str
    meeting_type: str
    case_id: Optional[str] = None
    case_match_status: str = "pending"
    participants: list[str] = []
    audio_filename: str = ""
    duration_seconds: float = 0.0
    state: str = "uploading"
    error_message: str = ""
    created_at: str = ""
    updated_at: str = ""


class MeetingSummaryResponse(BaseModel):
    id: str
    title: str
    meeting_type: str
    participants: list[str] = []
    transcript: str = ""
    segments: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    case_id: Optional[str] = None
    case_match_status: str = "pending"
    created_at: str = ""


class SearchResponse(BaseModel):
    results: list[MeetingSummaryResponse]
    total: int


def _orm_to_response(m: MeetingORM) -> MeetingResponse:
    return MeetingResponse(
        id=m.id,
        title=m.title,
        meeting_type=m.meeting_type,
        case_id=m.case_id,
        case_match_status=m.case_match_status,
        participants=json.loads(m.participants),
        audio_filename=m.audio_filename,
        duration_seconds=m.duration_seconds,
        state=m.state,
        error_message=m.error_message,
        created_at=m.created_at.isoformat() if m.created_at else "",
        updated_at=m.updated_at.isoformat() if m.updated_at else "",
    )


def _orm_to_summary(m: MeetingORM) -> MeetingSummaryResponse:
    return MeetingSummaryResponse(
        id=m.id,
        title=m.title,
        meeting_type=m.meeting_type,
        participants=json.loads(m.participants),
        transcript=m.transcript,
        segments=json.loads(m.segments),
        summary=json.loads(m.summary),
        case_id=m.case_id,
        case_match_status=m.case_match_status,
        created_at=m.created_at.isoformat() if m.created_at else "",
    )


def _get_db() -> Session:
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


def _broadcast_progress(task_id: str, progress: TaskProgress) -> None:
    task_progress_store[task_id] = progress
    ws_list = task_ws_connections.get(task_id, [])
    dead: list[WebSocket] = []
    msg = json.dumps({
        "task_id": progress.task_id,
        "state": progress.state,
        "transcribed_seconds": progress.transcribed_seconds,
        "total_seconds": progress.total_seconds,
        "percent": round(progress.transcribed_seconds / progress.total_seconds * 100, 1) if progress.total_seconds > 0 else 0,
        "error": progress.error,
    })
    for ws in ws_list:
        try:
            asyncio.get_event_loop().create_task(ws.send_text(msg))
        except Exception:
            dead.append(ws)
    for d in dead:
        ws_list.remove(d)


async def _process_task(meeting_id: str) -> None:
    db = _get_db()
    try:
        meeting = db.query(MeetingORM).filter(MeetingORM.id == meeting_id).first()
        if not meeting:
            return

        progress = TaskProgress(task_id=meeting_id, state="transcribing", total_seconds=0.0)
        _broadcast_progress(meeting_id, progress)

        meeting.state = "transcribing"
        db.commit()

        audio_path = os.path.join(UPLOAD_DIR, meeting.audio_filename)
        if not os.path.exists(audio_path):
            meeting.state = "failed"
            meeting.error_message = "Audio file not found"
            db.commit()
            progress.state = "failed"
            progress.error = "Audio file not found"
            _broadcast_progress(meeting_id, progress)
            return

        existing_cases = []
        try:
            rows = db.execute(text("SELECT DISTINCT case_id FROM meetings WHERE case_id IS NOT NULL AND case_match_status = 'matched'")).fetchall()
            existing_cases = [{"case_id": r[0], "case_number": r[0]} for r in rows]
        except Exception:
            pass

        def on_progress(p: TaskProgress) -> None:
            _broadcast_progress(meeting_id, p)

        result = await pipeline.run(
            audio_path=audio_path,
            meeting_type=MeetingType(meeting.meeting_type),
            participants=json.loads(meeting.participants),
            case_background=None,
            on_progress=on_progress,
            existing_cases=existing_cases,
        )

        meeting.transcript = result["transcript"]
        meeting.segments = json.dumps(result["segments"], ensure_ascii=False)
        meeting.summary = json.dumps(result["summary"], ensure_ascii=False)
        meeting.case_id = result["summary"].get("case_id")
        meeting.case_match_status = result["summary"].get("case_match_status", "pending")
        meeting.state = "completed"
        meeting.updated_at = datetime.now()

        total_dur = sum(s["end"] - s["start"] for s in result["segments"])
        meeting.duration_seconds = total_dur

        db.commit()

        progress.state = "completed"
        progress.transcribed_seconds = progress.total_seconds
        _broadcast_progress(meeting_id, progress)

    except Exception as exc:
        db = _get_db()
        meeting = db.query(MeetingORM).filter(MeetingORM.id == meeting_id).first()
        if meeting:
            meeting.state = "failed"
            meeting.error_message = str(exc)
            meeting.updated_at = datetime.now()
            db.commit()
        progress = TaskProgress(task_id=meeting_id, state="failed", error=str(exc))
        _broadcast_progress(meeting_id, progress)
    finally:
        db.close()


async def _queue_worker() -> None:
    while True:
        meeting_id = await task_queue.get()
        await _process_task(meeting_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(_queue_worker())
    yield
    worker_task.cancel()


app = FastAPI(title="会议录音转写与智能纪要生成API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/upload", response_model=MeetingResponse)
async def upload_audio(
    file: UploadFile = File(...),
    title: str = "",
    meeting_type: str = "client_consultation",
    participants: str = "[]",
    case_background: Optional[str] = None,
):
    allowed_ext = {".mp3", ".wav", ".m4a"}
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {allowed_ext}")

    meeting_id = str(uuid.uuid4())
    filename = f"{meeting_id}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    try:
        meeting_type_enum = MeetingType(meeting_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid meeting_type: {meeting_type}")

    async with aiofiles.open(filepath, "wb") as f:
        content = await file.read()
        await f.write(content)

    participants_list = json.loads(participants) if isinstance(participants, str) else participants

    db = _get_db()
    meeting = MeetingORM(
        id=meeting_id,
        title=title or file.filename or "未命名会议",
        meeting_type=meeting_type_enum.value,
        participants=json.dumps(participants_list, ensure_ascii=False),
        audio_filename=filename,
        state="queued",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)

    progress = TaskProgress(task_id=meeting_id, state="queued", total_seconds=0.0)
    _broadcast_progress(meeting_id, progress)

    await task_queue.put(meeting_id)

    db.close()
    return _orm_to_response(meeting)


@app.post("/api/upload/chunk", response_model=dict)
async def upload_chunk(
    file: UploadFile = File(...),
    upload_id: str = "",
    chunk_index: int = 0,
    total_chunks: int = 1,
    filename: str = "audio.wav",
    title: str = "",
    meeting_type: str = "client_consultation",
    participants: str = "[]",
):
    if not upload_id:
        upload_id = str(uuid.uuid4())

    chunk_dir = os.path.join(CHUNK_DIR, upload_id)
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:06d}")
    async with aiofiles.open(chunk_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    manifest_path = os.path.join(chunk_dir, "manifest.json")
    manifest: dict[str, Any] = {}
    if os.path.exists(manifest_path):
        async with aiofiles.open(manifest_path, "r") as f:
            manifest = json.loads(await f.read())
    else:
        manifest = {
            "upload_id": upload_id,
            "total_chunks": total_chunks,
            "received": [],
            "filename": filename,
            "title": title,
            "meeting_type": meeting_type,
            "participants": participants,
        }

    if chunk_index not in manifest["received"]:
        manifest["received"].append(chunk_index)

    async with aiofiles.open(manifest_path, "w") as f:
        await f.write(json.dumps(manifest))

    if len(manifest["received"]) >= total_chunks:
        meeting_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower()
        final_filename = f"{meeting_id}{ext}"
        final_path = os.path.join(UPLOAD_DIR, final_filename)

        async with aiofiles.open(final_path, "wb") as outf:
            for i in range(total_chunks):
                cp = os.path.join(chunk_dir, f"chunk_{i:06d}")
                async with aiofiles.open(cp, "rb") as inf:
                    await outf.write(await inf.read())

        import shutil
        shutil.rmtree(chunk_dir, ignore_errors=True)

        try:
            meeting_type_enum = MeetingType(meeting_type)
        except ValueError:
            meeting_type_enum = MeetingType.CLIENT_CONSULTATION

        participants_list = json.loads(participants) if isinstance(participants, str) else participants

        db = _get_db()
        meeting = MeetingORM(
            id=meeting_id,
            title=title or filename or "未命名会议",
            meeting_type=meeting_type_enum.value,
            participants=json.dumps(participants_list, ensure_ascii=False),
            audio_filename=final_filename,
            state="queued",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        db.add(meeting)
        db.commit()
        db.refresh(meeting)

        progress = TaskProgress(task_id=meeting_id, state="queued")
        _broadcast_progress(meeting_id, progress)

        await task_queue.put(meeting_id)
        db.close()

        return {"status": "complete", "meeting_id": meeting_id, "upload_id": upload_id}

    return {"status": "partial", "upload_id": upload_id, "received_chunks": len(manifest["received"]), "total_chunks": total_chunks}


@app.get("/api/meeting/{meeting_id}", response_model=MeetingResponse)
async def get_meeting(meeting_id: str):
    db = _get_db()
    meeting = db.query(MeetingORM).filter(MeetingORM.id == meeting_id).first()
    db.close()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return _orm_to_response(meeting)


@app.get("/api/meeting/{meeting_id}/summary", response_model=MeetingSummaryResponse)
async def get_meeting_summary(meeting_id: str):
    db = _get_db()
    meeting = db.query(MeetingORM).filter(MeetingORM.id == meeting_id).first()
    db.close()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if meeting.state != "completed":
        raise HTTPException(status_code=400, detail=f"Meeting is in state: {meeting.state}, not completed yet")
    return _orm_to_summary(meeting)


@app.get("/api/meetings", response_model=SearchResponse)
async def search_meetings(
    case_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    participant: Optional[str] = None,
    keyword: Optional[str] = None,
    meeting_type: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    db = _get_db()
    query = db.query(MeetingORM)

    if case_id:
        query = query.filter(MeetingORM.case_id == case_id)
    if meeting_type:
        query = query.filter(MeetingORM.meeting_type == meeting_type)
    if date_from:
        query = query.filter(MeetingORM.created_at >= datetime.fromisoformat(date_from))
    if date_to:
        query = query.filter(MeetingORM.created_at <= datetime.fromisoformat(date_to))
    if participant:
        query = query.filter(MeetingORM.participants.contains(participant))
    if keyword:
        query = query.filter(
            (MeetingORM.transcript.contains(keyword)) | (MeetingORM.summary.contains(keyword))
        )

    total = query.count()
    results = query.order_by(MeetingORM.created_at.desc()).offset(skip).limit(limit).all()
    db.close()

    return SearchResponse(
        results=[_orm_to_summary(m) for m in results],
        total=total,
    )


@app.get("/api/meeting/{meeting_id}/search")
async def search_in_meeting(meeting_id: str, keyword: str = Query(..., min_length=1)):
    db = _get_db()
    meeting = db.query(MeetingORM).filter(MeetingORM.id == meeting_id).first()
    db.close()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    segments = json.loads(meeting.segments)
    hits = []
    for seg in segments:
        if keyword.lower() in seg.get("text", "").lower():
            hits.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": seg.get("speaker", ""),
            })

    return {"meeting_id": meeting_id, "keyword": keyword, "hits": hits, "total_hits": len(hits)}


@app.websocket("/ws/task/{task_id}")
async def websocket_task_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    if task_id not in task_ws_connections:
        task_ws_connections[task_id] = []
    task_ws_connections[task_id].append(websocket)

    if task_id in task_progress_store:
        progress = task_progress_store[task_id]
        msg = json.dumps({
            "task_id": progress.task_id,
            "state": progress.state,
            "transcribed_seconds": progress.transcribed_seconds,
            "total_seconds": progress.total_seconds,
            "percent": round(progress.transcribed_seconds / progress.total_seconds * 100, 1) if progress.total_seconds > 0 else 0,
            "error": progress.error,
        })
        await websocket.send_text(msg)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if task_id in task_ws_connections:
            task_ws_connections[task_id].remove(websocket)
            if not task_ws_connections[task_id]:
                del task_ws_connections[task_id]


@app.get("/api/meeting/{meeting_id}/export/word")
async def export_word(meeting_id: str):
    db = _get_db()
    meeting = db.query(MeetingORM).filter(MeetingORM.id == meeting_id).first()
    db.close()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if meeting.state != "completed":
        raise HTTPException(status_code=400, detail="Meeting not completed yet")

    from docx import Document
    from docx.shared import Pt, Inches

    doc = Document()
    doc.add_heading("会议纪要", level=0)

    info_table = doc.add_table(rows=5, cols=2)
    info_data = [
        ("会议主题", meeting.title),
        ("会议类型", meeting.meeting_type),
        ("参会人员", ", ".join(json.loads(meeting.participants))),
        ("会议时间", meeting.created_at.isoformat() if meeting.created_at else ""),
        ("案件编号", meeting.case_id or "待关联"),
    ]
    for i, (label, value) in enumerate(info_data):
        info_table.rows[i].cells[0].text = label
        info_table.rows[i].cells[1].text = value

    doc.add_heading("转写文本", level=1)
    doc.add_paragraph(meeting.transcript)

    summary_data = json.loads(meeting.summary)
    doc.add_heading("智能纪要", level=1)
    doc.add_paragraph(summary_data.get("raw_summary", ""))

    export_dir = os.path.join(UPLOAD_DIR, "exports")
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, f"{meeting_id}.docx")
    doc.save(out_path)

    return FileResponse(out_path, filename=f"{meeting.title or '会议纪要'}.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@app.get("/api/meeting/{meeting_id}/export/pdf")
async def export_pdf(meeting_id: str):
    db = _get_db()
    meeting = db.query(MeetingORM).filter(MeetingORM.id == meeting_id).first()
    db.close()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if meeting.state != "completed":
        raise HTTPException(status_code=400, detail="Meeting not completed yet")

    summary_data = json.loads(meeting.summary)
    html_content = f"""
    <html><head><meta charset="utf-8"><style>
    body {{ font-family: SimSun, serif; margin: 40px; }}
    h1 {{ text-align: center; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    td, th {{ border: 1px solid #000; padding: 8px; }}
    h2 {{ border-bottom: 2px solid #333; padding-bottom: 5px; }}
    </style></head><body>
    <h1>会议纪要</h1>
    <table>
    <tr><td><strong>会议主题</strong></td><td>{meeting.title}</td></tr>
    <tr><td><strong>会议类型</strong></td><td>{meeting.meeting_type}</td></tr>
    <tr><td><strong>参会人员</strong></td><td>{', '.join(json.loads(meeting.participants))}</td></tr>
    <tr><td><strong>会议时间</strong></td><td>{meeting.created_at.isoformat() if meeting.created_at else ''}</td></tr>
    <tr><td><strong>案件编号</strong></td><td>{meeting.case_id or '待关联'}</td></tr>
    </table>
    <h2>转写文本</h2>
    <pre style="white-space: pre-wrap;">{meeting.transcript}</pre>
    <h2>智能纪要</h2>
    <pre style="white-space: pre-wrap;">{summary_data.get('raw_summary', '')}</pre>
    </body></html>
    """

    from weasyprint import HTML
    export_dir = os.path.join(UPLOAD_DIR, "exports")
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, f"{meeting_id}.pdf")
    HTML(string=html_content).write_pdf(out_path)

    return FileResponse(out_path, filename=f"{meeting.title or '会议纪要'}.pdf", media_type="application/pdf")


@app.put("/api/meeting/{meeting_id}/case")
async def update_meeting_case(meeting_id: str, case_id: str = "", case_match_status: str = "matched"):
    db = _get_db()
    meeting = db.query(MeetingORM).filter(MeetingORM.id == meeting_id).first()
    if not meeting:
        db.close()
        raise HTTPException(status_code=404, detail="Meeting not found")
    meeting.case_id = case_id
    meeting.case_match_status = case_match_status
    meeting.updated_at = datetime.now()
    db.commit()
    db.close()
    return {"message": "Case updated", "meeting_id": meeting_id, "case_id": case_id}


@app.put("/api/meeting/{meeting_id}/transcript")
async def update_meeting_transcript(meeting_id: str, transcript: str = "", summary_raw: str = ""):
    db = _get_db()
    meeting = db.query(MeetingORM).filter(MeetingORM.id == meeting_id).first()
    if not meeting:
        db.close()
        raise HTTPException(status_code=404, detail="Meeting not found")
    if transcript:
        meeting.transcript = transcript
    if summary_raw:
        current = json.loads(meeting.summary)
        current["raw_summary"] = summary_raw
        meeting.summary = json.dumps(current, ensure_ascii=False)
    meeting.updated_at = datetime.now()
    db.commit()
    db.close()
    return {"message": "Transcript updated", "meeting_id": meeting_id}


@app.get("/api/templates", response_model=list[TemplateRecord])
async def list_templates(meeting_type: Optional[str] = None):
    mt = MeetingType(meeting_type) if meeting_type else None
    return template_manager.list_all(mt)


@app.post("/api/templates", response_model=TemplateRecord)
async def create_template(req: TemplateCreateRequest):
    try:
        return template_manager.create(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/templates/{template_id}", response_model=TemplateRecord)
async def update_template(template_id: str, req: TemplateUpdateRequest):
    result = template_manager.update(template_id, req)
    if result is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return result


@app.delete("/api/templates/{template_id}")
async def delete_template(template_id: str):
    if not template_manager.delete(template_id):
        raise HTTPException(status_code=404, detail="Template not found")
    return {"message": "Template deleted"}


@app.get("/api/templates/{template_id}/versions/{version}")
async def get_template_version(template_id: str, version: int):
    record = template_manager.get_version(template_id, version)
    if record is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return record


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
