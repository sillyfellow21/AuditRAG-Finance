from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from pypdf import PdfReader
import pytesseract

from core.config import Settings
from core.models import ParsedDocument
from monitoring.observability import Observability


class FinancialDocumentParser:
    SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

    def __init__(self, settings: Settings, observability: Observability) -> None:
        self.settings = settings
        self.observability = observability
        self._docling_converter: Optional[object] = None
        self._init_docling()

        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    def _init_docling(self) -> None:
        if not self.settings.docling_enabled:
            self._docling_converter = None
            self.observability.log_event(
                "docling.initialized",
                {"status": "disabled_by_config"},
            )
            return

        try:
            from docling.document_converter import DocumentConverter

            self._docling_converter = DocumentConverter()
            self.observability.log_event("docling.initialized", {"status": "ok"})
        except Exception as exc:
            self._docling_converter = None
            self.observability.log_warning(
                "docling.initialized",
                {
                    "status": "fallback_enabled",
                    "reason": str(exc),
                },
            )

    def parse(self, document_id: str, file_path: Path) -> ParsedDocument:
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")

        parse_warnings: List[str] = []
        parser_used = "unknown"
        text = ""
        pages: List[str] = []

        should_try_docling = (
            self._docling_converter is not None
            and (extension != ".pdf" or self.settings.docling_for_pdf)
        )
        if should_try_docling:
            try:
                text, pages = self._parse_with_docling(file_path)
                if text.strip():
                    parser_used = "docling"
            except Exception as exc:
                parse_warnings.append(f"Docling parser failed: {exc}")
                self.observability.log_warning(
                    "parsing.docling_failed",
                    {"file": file_path.name, "error": str(exc)},
                )

        if not text.strip():
            if extension == ".pdf":
                text, pages = self._parse_pdf(file_path)
                parser_used = "pypdf"
            else:
                text = self._parse_image(file_path)
                pages = [text] if text else []
                parser_used = "pytesseract"

        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            parse_warnings.append("No readable text found in document.")

        return ParsedDocument(
            document_id=document_id,
            filename=file_path.name,
            file_type=extension,
            text=cleaned_text,
            pages=pages,
            parser_used=parser_used,
            parse_warnings=parse_warnings,
        )

    def _parse_with_docling(self, file_path: Path) -> Tuple[str, List[str]]:
        if self._docling_converter is None:
            return "", []

        conversion = self._docling_converter.convert(str(file_path))
        document = getattr(conversion, "document", None)
        if document is None:
            return "", []

        text = ""
        if hasattr(document, "export_to_markdown"):
            text = document.export_to_markdown() or ""
        if not text and hasattr(document, "export_to_text"):
            text = document.export_to_text() or ""
        if not text and hasattr(document, "export_to_dict"):
            exported = document.export_to_dict()
            text = json.dumps(exported, ensure_ascii=True)

        pages: List[str] = []
        raw_pages = getattr(conversion, "pages", None)
        if isinstance(raw_pages, list):
            for page in raw_pages:
                page_text = getattr(page, "text", "")
                if isinstance(page_text, str) and page_text.strip():
                    pages.append(self._clean_text(page_text))

        if not pages and text:
            pages = [self._clean_text(text)]

        return text, pages

    def _parse_pdf(self, file_path: Path) -> Tuple[str, List[str]]:
        reader = PdfReader(str(file_path))
        pages: List[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            pages.append(self._clean_text(extracted))

        text = "\n\n".join([page for page in pages if page])
        return text, pages

    def _parse_image(self, file_path: Path) -> str:
        with Image.open(file_path) as img:
            text = pytesseract.image_to_string(img)
        return self._clean_text(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        normalized = text.replace("\r", "\n")
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"[ \t]{2,}", " ", normalized)
        return normalized.strip()
