"""Mistral OCR preparation for the shared PDF message contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, TYPE_CHECKING

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llm_registry.llm_registry import MeteredOperationSpec
from llm_api_adapter.models.messages.chat_message import Message, Messages, UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart

if TYPE_CHECKING:
    from .adapter import MistralAdapter


MISTRAL_OCR_MODEL = "mistral-ocr-4-1"


@dataclass(frozen=True)
class PreparedOCROperation:
    """Verified OCR inputs retained until response cost finalization."""

    model: str
    pages_processed: Optional[int]
    meter: Optional[MeteredOperationSpec]


@dataclass(frozen=True)
class PreparedDocuments:
    """Mistral-ready messages and separate OCR usage for one request."""

    messages: Messages
    ocr_operations: tuple[PreparedOCROperation, ...] = ()


def prepare_document_messages(
    adapter: "MistralAdapter", messages: list[Message] | Messages, timeout_s: Optional[float]
) -> PreparedDocuments:
    normalized = adapter._normalize_messages(messages)
    documents = document_parts(normalized)
    if not documents:
        return PreparedDocuments(messages=normalized)

    meter = adapter._ocr_meter()
    markdowns: list[str] = []
    ocr_operations: list[PreparedOCROperation] = []
    for document in documents:
        response = adapter._post_ocr_payload(build_ocr_payload(document), timeout_s)
        markdowns.append(extract_ocr_markdown(response))
        ocr_operations.append(extract_ocr_operation(response, meter))

    return PreparedDocuments(
        messages=replace_documents(normalized, markdowns),
        ocr_operations=tuple(ocr_operations),
    )


async def prepare_document_messages_async(
    adapter: "MistralAdapter", messages: list[Message] | Messages, timeout_s: Optional[float]
) -> Messages:
    normalized = adapter._normalize_messages(messages)
    documents = document_parts(normalized)
    if not documents:
        return normalized
    markdowns = [
        extract_ocr_markdown(await adapter._apost_ocr_payload(build_ocr_payload(document), timeout_s))
        for document in documents
    ]
    return replace_documents(normalized, markdowns)


def document_parts(messages: Messages) -> list[DocumentPart]:
    return [file for message in messages.items if isinstance(message, UserMessage) and message.files for file in message.files if isinstance(file, DocumentPart)]


def replace_documents(messages: Messages, markdowns: list[str]) -> Messages:
    iterator = iter(markdowns)
    processed: list[Message] = []
    for message in messages.items:
        if not isinstance(message, UserMessage) or not message.files:
            processed.append(message)
            continue
        document_markdowns = [next(iterator) for file in message.files if isinstance(file, DocumentPart)]
        if not document_markdowns:
            processed.append(message)
            continue
        files = [file for file in message.files if not isinstance(file, DocumentPart)]
        processed.append(UserMessage(content=append_document_markdown(message.content, document_markdowns), files=files or None))
    return Messages(processed)


def append_document_markdown(content: str, markdowns: list[str]) -> str:
    documents = [f'<document index="{index}">\n{markdown}\n</document>' for index, markdown in enumerate(markdowns, start=1)]
    return "\n\n".join([content, *documents])


def build_ocr_payload(document: DocumentPart) -> dict[str, Any]:
    document_url = document.url if document._is_url() else document._to_data_uri()
    return {"model": MISTRAL_OCR_MODEL, "document": {"type": "document_url", "document_url": document_url}}


def extract_ocr_markdown(response: Mapping[str, Any]) -> str:
    pages = response.get("pages")
    if not isinstance(pages, list):
        raise LLMAPIClientError(detail="Mistral OCR response does not contain document pages")
    markdowns = [page["markdown"] for page in pages if isinstance(page, Mapping) and isinstance(page.get("markdown"), str) and page["markdown"].strip()]
    if not markdowns:
        raise LLMAPIClientError(detail="Mistral OCR response contains no document text")
    return "\n\n---\n\n".join(markdowns)


def extract_ocr_operation(
    response: Mapping[str, Any],
    meter: Optional[MeteredOperationSpec],
) -> PreparedOCROperation:
    """Read Mistral's authoritative page count without inventing usage."""
    usage_info = response.get("usage_info")
    pages_processed = (
        usage_info.get("pages_processed")
        if isinstance(usage_info, Mapping)
        else None
    )
    if (
        isinstance(pages_processed, bool)
        or not isinstance(pages_processed, int)
        or pages_processed < 0
    ):
        pages_processed = None
    return PreparedOCROperation(
        model=MISTRAL_OCR_MODEL,
        pages_processed=pages_processed,
        meter=meter,
    )
