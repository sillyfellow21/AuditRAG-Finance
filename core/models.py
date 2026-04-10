from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None
    evidence: Optional[str] = None


class ExtractedFields(BaseModel):
    vendor: Optional[str] = None
    date: Optional[str] = None
    total: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    raw_fields: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    document_id: str
    filename: str
    file_type: str
    text: str
    pages: List[str] = Field(default_factory=list)
    parser_used: str
    parse_warnings: List[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnomalySignal(BaseModel):
    code: str
    severity: str
    message: str
    metric: Optional[float] = None


class AnomalyAssessment(BaseModel):
    is_suspicious: bool
    risk_level: str
    score: float = Field(ge=0.0, le=1.0)
    signals: List[AnomalySignal] = Field(default_factory=list)


class ExplainabilityDetails(BaseModel):
    decision_path: List[str] = Field(default_factory=list)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)
    anomaly: Optional[AnomalyAssessment] = None
    notes: List[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    parser_used: str
    extracted_data: ExtractedFields
    indexed_chunks: int
    summary: str
    anomaly: Optional[AnomalyAssessment] = None


class AskRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=3, max_length=2000)


class ErrorResponse(BaseModel):
    error: str


class AnswerResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: Literal["charge_explanation", "anomaly", "summary"]
    evidence: List[str] = Field(default_factory=list)
    flags: List[Literal["duplicate", "suspicious", "recurring"]] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    structured_data: Optional[ExtractedFields] = None
    anomaly: Optional[AnomalyAssessment] = None
    explainability: ExplainabilityDetails = Field(default_factory=ExplainabilityDetails)
    reasoning: str


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    file_path: str
    text_path: str
    uploaded_at: datetime
    parser_used: str
    extracted_data: ExtractedFields
    anomaly: Optional[AnomalyAssessment] = None
    chunks: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditLogEntry(BaseModel):
    event_id: str
    timestamp: datetime
    actor: str
    action: str
    resource: str
    outcome: str
    request_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class AuditLogResponse(BaseModel):
    count: int
    logs: List[AuditLogEntry] = Field(default_factory=list)
