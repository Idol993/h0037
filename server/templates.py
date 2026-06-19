from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from jinja2 import Environment, BaseLoader, TemplateSyntaxError
from pydantic import BaseModel, Field


class MeetingType(str, Enum):
    CLIENT_CONSULTATION = "client_consultation"
    INTERNAL_DISCUSSION = "internal_discussion"
    PRE_TRIAL_CONFERENCE = "pre_trial_conference"


class TemplateCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    meeting_type: MeetingType
    content: str = Field(..., min_length=1)
    description: Optional[str] = None


class TemplateUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    description: Optional[str] = None


class TemplateVersionRecord(BaseModel):
    version: int
    content: str
    changed_at: str
    changed_by: Optional[str] = None


class TemplateRecord(BaseModel):
    id: str
    name: str
    meeting_type: MeetingType
    content: str
    description: Optional[str] = None
    is_active: bool = True
    current_version: int = 1
    versions: list[TemplateVersionRecord] = []
    created_at: str
    updated_at: str


CLIENT_CONSULTATION_TEMPLATE = """你是一位资深法律助理，正在整理一场律师与客户之间的咨询会议纪要。

## 会议信息
- 会议类型：客户咨询
- 参会人员：{{ participants | join('、') }}
{% if case_background %}- 案件背景：{{ case_background }}{% endif %}

## 转写文本
{{ transcript }}

---

请按以下格式输出结构化纪要：

### 一、会议主题
（一句话概括本次咨询的核心议题）

### 二、客户诉求
（逐条列出客户提出的诉求和关注点，标注对应时间戳）
{% raw %}
1. [HH:MM:SS] 诉求内容
{% endraw %}

### 三、法律建议
（律师给出的法律分析和建议，标注时间戳）
{% raw %}
1. [HH:MM:SS] 建议内容
{% endraw %}

### 四、关键事实陈述
（客户陈述的重要事实，可能影响案件走向）

### 五、待办事项
| 序号 | 事项 | 负责人 | 截止日期 |
|------|------|--------|----------|
{% raw %}
| 1 | ... | ... | ... |
{% endraw %}

### 六、风险提示
（可能存在的法律风险或需要注意的问题）

### 七、下次沟通要点
（后续需要进一步确认或跟进的事项）
"""

INTERNAL_DISCUSSION_TEMPLATE = """你是一位资深法律助理，正在整理一场律所内部的案件讨论会议纪要。

## 会议信息
- 会议类型：内部讨论
- 参会人员：{{ participants | join('、') }}
{% if case_background %}- 案件背景：{{ case_background }}{% endif %}

## 转写文本
{{ transcript }}

---

请按以下格式输出结构化纪要：

### 一、会议主题
（一句话概括本次讨论的核心议题）

### 二、讨论要点
（按讨论顺序列出核心观点，标注时间戳和发言人）
{% raw %}
1. [HH:MM:SS] 发言人：观点内容
{% endraw %}

### 三、决策事项
（会议中形成的明确决策，标注时间戳）
{% raw %}
1. [HH:MM:SS] 决策内容
{% endraw %}

### 四、待办事项
| 序号 | 事项 | 负责人 | 截止日期 | 优先级 |
|------|------|--------|----------|--------|
{% raw %}
| 1 | ... | ... | ... | 高/中/低 |
{% endraw %}

### 五、分歧意见
（讨论中未达成一致的观点及各方立场）

### 六、法律依据与判例引用
（讨论中引用的法律法规和参考判例）

### 七、下一步行动计划
（明确的时间节点和责任人）
"""

PRE_TRIAL_CONFERENCE_TEMPLATE = """你是一位资深法律助理，正在整理一场庭前准备会议纪要。

## 会议信息
- 会议类型：庭前会议
- 参会人员：{{ participants | join('、') }}
{% if case_background %}- 案件背景：{{ case_background }}{% endif %}

## 转写文本
{{ transcript }}

---

请按以下格式输出结构化纪要：

### 一、案件概述
（案由、当事人、管辖法院等基本信息）

### 二、庭审策略
（出庭律师讨论的庭审策略和应对方案，标注时间戳）
{% raw %}
1. [HH:MM:SS] 策略要点
{% endraw %}

### 三、证据梳理
| 序号 | 证据名称 | 证明目的 | 举证顺序 | 备注 |
|------|----------|----------|----------|------|
{% raw %}
| 1 | ... | ... | ... | ... |
{% endraw %}

### 四、对方可能抗辩及应对
（预判对方律师可能的抗辩理由，以及我方应对策略）

### 五、争议焦点
（案件核心争议点及我方论证思路）

### 六、庭审分工
| 律师 | 职责 | 关注要点 |
|------|------|----------|
{% raw %}
| ... | ... | ... |
{% endraw %}

### 七、风险提示
（庭审中可能面临的不利情况及预案）

### 八、待办事项
| 序号 | 事项 | 负责人 | 截止日期 |
|------|------|--------|----------|
{% raw %}
| 1 | ... | ... | ... |
{% endraw %}
"""


_jinja_env = Environment(loader=BaseLoader())


def _now_iso() -> str:
    return datetime.now().isoformat()


class TemplateManager:
    def __init__(self) -> None:
        self._templates: dict[str, TemplateRecord] = {}
        self._init_builtins()

    def _init_builtins(self) -> None:
        builtins = [
            ("客户咨询纪要模板", MeetingType.CLIENT_CONSULTATION, CLIENT_CONSULTATION_TEMPLATE, "适用于律师与客户的咨询会议，侧重客户诉求和法律建议"),
            ("内部讨论纪要模板", MeetingType.INTERNAL_DISCUSSION, INTERNAL_DISCUSSION_TEMPLATE, "适用于律所内部案件讨论，侧重决策和待办事项"),
            ("庭前会议纪要模板", MeetingType.PRE_TRIAL_CONFERENCE, PRE_TRIAL_CONFERENCE_TEMPLATE, "适用于庭前准备会议，侧重策略和证据梳理"),
        ]
        for name, mtype, content, desc in builtins:
            tid = str(uuid.uuid4())
            now = _now_iso()
            self._templates[tid] = TemplateRecord(
                id=tid,
                name=name,
                meeting_type=mtype,
                content=content,
                description=desc,
                is_active=True,
                current_version=1,
                versions=[TemplateVersionRecord(version=1, content=content, changed_at=now)],
                created_at=now,
                updated_at=now,
            )

    def create(self, req: TemplateCreateRequest) -> TemplateRecord:
        tid = str(uuid.uuid4())
        now = _now_iso()
        self._validate_template(req.content)
        record = TemplateRecord(
            id=tid,
            name=req.name,
            meeting_type=req.meeting_type,
            content=req.content,
            description=req.description,
            is_active=True,
            current_version=1,
            versions=[TemplateVersionRecord(version=1, content=req.content, changed_at=now)],
            created_at=now,
            updated_at=now,
        )
        self._templates[tid] = record
        return record

    def update(self, template_id: str, req: TemplateUpdateRequest, changed_by: Optional[str] = None) -> Optional[TemplateRecord]:
        record = self._templates.get(template_id)
        if record is None:
            return None
        if req.name is not None:
            record.name = req.name
        if req.description is not None:
            record.description = req.description
        if req.content is not None:
            self._validate_template(req.content)
            record.current_version += 1
            record.versions.append(
                TemplateVersionRecord(
                    version=record.current_version,
                    content=req.content,
                    changed_at=_now_iso(),
                    changed_by=changed_by,
                )
            )
            record.content = req.content
        record.updated_at = _now_iso()
        return record

    def delete(self, template_id: str) -> bool:
        return self._templates.pop(template_id, None) is not None

    def get(self, template_id: str) -> Optional[TemplateRecord]:
        return self._templates.get(template_id)

    def list_all(self, meeting_type: Optional[MeetingType] = None) -> list[TemplateRecord]:
        results = list(self._templates.values())
        if meeting_type is not None:
            results = [t for t in results if t.meeting_type == meeting_type]
        return results

    def get_by_meeting_type(self, meeting_type: MeetingType) -> Optional[TemplateRecord]:
        for t in self._templates.values():
            if t.meeting_type == meeting_type and t.is_active:
                return t
        return None

    def render_prompt(
        self,
        meeting_type: MeetingType,
        transcript: str,
        participants: Optional[list[str]] = None,
        case_background: Optional[str] = None,
        template_id: Optional[str] = None,
    ) -> str:
        if template_id:
            record = self._templates.get(template_id)
            if record is None:
                raise ValueError(f"Template {template_id} not found")
        else:
            record = self.get_by_meeting_type(meeting_type)
            if record is None:
                raise ValueError(f"No active template for meeting type {meeting_type}")

        template = _jinja_env.from_string(record.content)
        return template.render(
            participants=participants or [],
            case_background=case_background or "",
            transcript=transcript,
        )

    def get_version(self, template_id: str, version: int) -> Optional[TemplateVersionRecord]:
        record = self._templates.get(template_id)
        if record is None:
            return None
        for v in record.versions:
            if v.version == version:
                return v
        return None

    @staticmethod
    def _validate_template(content: str) -> None:
        try:
            _jinja_env.parse(content)
        except TemplateSyntaxError as exc:
            raise ValueError(f"Invalid Jinja2 template syntax: {exc}") from exc
