from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from openai import OpenAI

from .templates import MeetingType, TemplateManager


SLICE_DURATION_SECONDS = 1800
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = ""


@dataclass
class KeyDiscussionPoint:
    timestamp: str
    content: str


@dataclass
class TodoItem:
    item: str
    assignee: str = ""
    deadline: str = ""


@dataclass
class SummaryResult:
    topic: str = ""
    key_points: list[KeyDiscussionPoint] = field(default_factory=list)
    todos: list[TodoItem] = field(default_factory=list)
    client_demands: list[str] = field(default_factory=list)
    risk_alerts: list[str] = field(default_factory=list)
    raw_summary: str = ""
    case_id: Optional[str] = None
    case_match_status: str = "pending"
    candidate_case_numbers: list[str] = field(default_factory=list)


@dataclass
class TaskProgress:
    task_id: str = ""
    state: str = "queued"
    transcribed_seconds: float = 0.0
    total_seconds: float = 0.0
    error: str = ""


def preprocess_audio(input_path: str, output_dir: str) -> list[str]:
    converted = os.path.join(output_dir, "converted.wav")
    cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-f", "wav",
        "-y", converted,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    probe_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", converted]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())

    if duration <= SLICE_DURATION_SECONDS:
        return [converted]

    slices: list[str] = []
    idx = 0
    start = 0.0
    while start < duration:
        out_path = os.path.join(output_dir, f"slice_{idx:04d}.wav")
        cmd = [
            "ffmpeg", "-i", converted,
            "-ss", str(start),
            "-t", str(SLICE_DURATION_SECONDS),
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", str(TARGET_CHANNELS),
            "-f", "wav",
            "-y", out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        slices.append(out_path)
        start += SLICE_DURATION_SECONDS
        idx += 1

    return slices


def transcribe_audio(
    audio_path: str,
    model_size: str = "medium",
    language: str = "zh",
    on_progress: Optional[Callable[[float, float], None]] = None,
) -> list[TranscriptSegment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    probe_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    total_duration = float(probe_result.stdout.strip())

    segments, info = model.transcribe(audio_path, language=language, vad_filter=True)

    results: list[TranscriptSegment] = []
    for seg in segments:
        results.append(TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip()))
        if on_progress:
            on_progress(seg.end, total_duration)

    return results


def run_diarization(audio_path: str, num_speakers: Optional[int] = None) -> list[dict[str, Any]]:
    from pyannote.audio import Pipeline as PyannotePipeline

    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")
    pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

    diarization = pipeline(audio_path, num_speakers=num_speakers)

    turns: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    return turns


def merge_transcript_with_speakers(
    segments: list[TranscriptSegment],
    diarization_turns: list[dict[str, Any]],
) -> list[TranscriptSegment]:
    if not diarization_turns:
        for seg in segments:
            seg.speaker = "未知"
        return segments

    speaker_map: dict[str, str] = {}
    speaker_counter = 0
    for turn in diarization_turns:
        raw = turn["speaker"]
        if raw not in speaker_map:
            speaker_counter += 1
            if speaker_counter <= 1:
                speaker_map[raw] = "律师A"
            elif speaker_counter == 2:
                speaker_map[raw] = "客户B"
            else:
                speaker_map[raw] = f"发言人{speaker_counter}"

    for seg in segments:
        mid = (seg.start + seg.end) / 2.0
        best_turn = min(diarization_turns, key=lambda t: abs((t["start"] + t["end"]) / 2.0 - mid))
        seg.speaker = speaker_map.get(best_turn["speaker"], "未知")

    return segments


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for seg in segments:
        ts = format_timestamp(seg.start)
        lines.append(f"[{ts}] {seg.speaker}：{seg.text}")
    return "\n".join(lines)


def generate_summary(
    transcript_text: str,
    meeting_type: MeetingType,
    template_manager: TemplateManager,
    participants: Optional[list[str]] = None,
    case_background: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4o",
) -> SummaryResult:
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = openai_base_url or os.environ.get("OPENAI_BASE_URL", None)

    prompt = template_manager.render_prompt(
        meeting_type=meeting_type,
        transcript=transcript_text,
        participants=participants or [],
        case_background=case_background or "",
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": "你是一位专业的法律助理，擅长从会议录音转写文本中提取关键信息并生成结构化纪要。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    raw_summary = response.choices[0].message.content or ""

    result = SummaryResult(raw_summary=raw_summary)
    result.topic = _extract_field(raw_summary, "会议主题") or _extract_field(raw_summary, "案件概述") or ""
    result.key_points = _extract_key_points(raw_summary)
    result.todos = _extract_todos(raw_summary)
    result.client_demands = _extract_list_field(raw_summary, "客户诉求")
    result.risk_alerts = _extract_list_field(raw_summary, "风险提示")

    return result


def _extract_field(text: str, header: str) -> Optional[str]:
    patterns = [
        rf"###\s*(?:[一二三四五六七八九十]+[、.]\s*)?{re.escape(header)}\s*\n+(.*?)(?=\n###|\Z)",
        rf"##?\s*(?:[一二三四五六七八九十]+[、.]\s*)?{re.escape(header)}\s*\n+(.*?)(?=\n##?#\s|\Z)",
        rf"\*\*{re.escape(header)}\*\*\s*\n+(.*?)(?=\n\*\*|\Z)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return None


def _extract_list_field(text: str, header: str) -> list[str]:
    content = _extract_field(text, header)
    if not content:
        return []
    items: list[str] = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells and not all(c in {"序号", "事项", "负责人", "截止日期", "优先级", "备注", "-"} for c in cells):
                if not all(set(c) <= {"-"} for c in cells):
                    items.append(" ".join(cells))
            continue
        clean = line.lstrip("0123456789.-) \uff09\uff09")
        if clean and not clean.startswith("-"):
            items.append(clean)
        elif clean.startswith("-"):
            items.append(clean[1:].strip())
    return [item for item in items if item]


def _extract_key_points(text: str) -> list[KeyDiscussionPoint]:
    content = _extract_field(text, "讨论要点") or _extract_field(text, "庭审策略") or ""
    if not content:
        return []
    points: list[KeyDiscussionPoint] = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        ts_match = re.search(r"\[(\d{2}:\d{2}:\d{2})\]", line)
        ts = ts_match.group(1) if ts_match else ""
        clean = re.sub(r"^\d+[\.\)]\s*", "", line)
        clean = re.sub(r"\[\d{2}:\d{2}:\d{2}\]\s*", "", clean).strip()
        if clean:
            points.append(KeyDiscussionPoint(timestamp=ts, content=clean))
    return points


def _extract_todos(text: str) -> list[TodoItem]:
    content = _extract_field(text, "待办事项")
    if not content:
        return []
    items: list[TodoItem] = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if not cells:
                continue
            if all(c in {"序号", "事项", "负责人", "截止日期", "优先级", "备注", "-"} for c in cells):
                continue
            if all(set(c) <= {"-"} for c in cells):
                continue
            item_text = cells[0] if len(cells) >= 1 else ""
            assignee = cells[1] if len(cells) >= 2 else ""
            deadline = cells[2] if len(cells) >= 3 else ""
            if item_text:
                items.append(TodoItem(item=item_text, assignee=assignee, deadline=deadline))
            continue
        if line.startswith("-"):
            clean = line[1:].strip()
            if clean:
                items.append(TodoItem(item=clean))
            continue
        clean = re.sub(r"^[一二三四五六七八九十]+[、.]\s*", "", line)
        clean = re.sub(r"^\d+[\.\)]\s*", "", clean).strip()
        if clean:
            items.append(TodoItem(item=clean))
    return items


def match_case(summary_text: str, existing_cases: list[dict[str, str]]) -> tuple[Optional[str], str, list[str]]:
    mentions: list[str] = []

    p1 = re.compile(r"[（(]?\s*案号\s*[:：]?\s*([（(]?\d{4}[）)][\u4e00-\u9fff\d]+号?)[）)]?(?:\s|[,，。.；;]|$)")
    mentions.extend(p1.findall(summary_text))

    p2 = re.compile(r"[（(](\d{4})[）)][京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁]?[\d]+[\u4e00-\u9fff]*\d+号")
    mentions.extend(p2.findall(summary_text))

    p3 = re.compile(r"((?:20\d{2})[）)]?[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁]?[\d]+[\u4e00-\u9fff]*\d+号)")
    mentions.extend(p3.findall(summary_text))

    p4 = re.compile(r"((?:20\d{2})\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?\s*[^\s，。,;；]{2,20}号)")
    mentions.extend(p4.findall(summary_text))

    p5 = re.compile(r"[（(](\d{4}[）)][京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁]\d+[^）)\s]{0,30}号)")
    mentions.extend(p5.findall(summary_text))

    if not mentions:
        return None, "pending", []

    unique_mentions = list(dict.fromkeys(mentions))

    for mention in unique_mentions:
        for case in existing_cases:
            cn = case.get("case_number", "")
            if mention in cn or cn in mention:
                return case["case_id"], "matched", unique_mentions

    return None, "unmatched", unique_mentions


class Pipeline:
    def __init__(
        self,
        template_manager: TemplateManager,
        whisper_model_size: str = "medium",
        openai_model: str = "gpt-4o",
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        skip_diarization: bool = False,
    ) -> None:
        self.template_manager = template_manager
        self.whisper_model_size = whisper_model_size
        self.openai_model = openai_model
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.skip_diarization = skip_diarization

    async def run(
        self,
        audio_path: str,
        meeting_type: MeetingType,
        participants: Optional[list[str]] = None,
        case_background: Optional[str] = None,
        on_progress: Optional[Callable[[TaskProgress], None]] = None,
        existing_cases: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        progress = TaskProgress(task_id=str(uuid.uuid4()), state="transcribing", total_seconds=0.0)

        if on_progress:
            on_progress(progress)

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = asyncio.get_event_loop()
            slices = await loop.run_in_executor(None, preprocess_audio, audio_path, tmpdir)

            all_segments: list[TranscriptSegment] = []
            offset = 0.0

            for slice_path in slices:
                probe_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", slice_path]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                slice_duration = float(probe_result.stdout.strip())
                progress.total_seconds += slice_duration

            for slice_path in slices:
                def _progress_cb(current: float, total: float) -> None:
                    progress.transcribed_seconds = offset + current
                    if on_progress:
                        on_progress(progress)

                segs = await loop.run_in_executor(
                    None,
                    lambda p=slice_path: transcribe_audio(p, self.whisper_model_size, on_progress=_progress_cb),
                )

                for seg in segs:
                    seg.start += offset
                    seg.end += offset
                all_segments.extend(segs)

                probe_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", slice_path]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                offset += float(probe_result.stdout.strip())

            if not self.skip_diarization:
                try:
                    diarization_turns = await loop.run_in_executor(None, run_diarization, audio_path)
                    all_segments = merge_transcript_with_speakers(all_segments, diarization_turns)
                except Exception:
                    for seg in all_segments:
                        seg.speaker = "未知"

        progress.state = "summarizing"
        progress.transcribed_seconds = progress.total_seconds
        if on_progress:
            on_progress(progress)

        transcript_text = segments_to_text(all_segments)
        segments_data = [
            {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
            for s in all_segments
        ]

        summary = await loop.run_in_executor(
            None,
            lambda: generate_summary(
                transcript_text,
                meeting_type,
                self.template_manager,
                participants=participants,
                case_background=case_background,
                openai_base_url=self.openai_base_url,
                openai_api_key=self.openai_api_key,
                openai_model=self.openai_model,
            ),
        )

        case_id, match_status, candidates = match_case(summary.raw_summary, existing_cases or [])
        summary.case_id = case_id
        summary.case_match_status = match_status
        summary.candidate_case_numbers = candidates

        progress.state = "completed"
        if on_progress:
            on_progress(progress)

        return {
            "transcript": transcript_text,
            "segments": segments_data,
            "summary": {
                "topic": summary.topic,
                "key_points": [{"timestamp": p.timestamp, "content": p.content} for p in summary.key_points],
                "todos": [{"item": t.item, "assignee": t.assignee, "deadline": t.deadline} for t in summary.todos],
                "client_demands": summary.client_demands,
                "risk_alerts": summary.risk_alerts,
                "raw_summary": summary.raw_summary,
                "case_id": summary.case_id,
                "case_match_status": summary.case_match_status,
                "candidate_case_numbers": summary.candidate_case_numbers,
            },
        }
