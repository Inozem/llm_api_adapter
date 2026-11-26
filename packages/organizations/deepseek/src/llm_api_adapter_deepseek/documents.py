"""Private validation and slot helpers for DeepSeek PDF uploads."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, replace
from typing import Any, Mapping

from llm_api_adapter.models.messages.chat_message import Messages, UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart


MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_PDFS_PER_REQUEST = 20


@dataclass(frozen=True)
class PDFSlot:
    """One ordered PDF attachment owned by a single invocation."""

    message_index: int
    file_index: int
    content: bytes


def decode_pdf_part(part: DocumentPart) -> bytes:
    """Return validated PDF bytes from an inline part, rejecting remote URLs."""
    media_type = (part._get_media_type() or "").lower()
    if media_type != "application/pdf":
        raise ValueError("DeepSeek PDF media_type must be application/pdf")
    if part.data is not None:
        content = part.data
    elif isinstance(part.url, str) and part.url.startswith("data:"):
        prefix, separator, encoded = part.url.partition(",")
        metadata = prefix[5:].split(";") if prefix.lower().startswith("data:") else []
        if (
            not separator
            or not metadata
            or metadata[0].lower() != "application/pdf"
            or "base64" not in {value.lower() for value in metadata[1:]}
        ):
            raise ValueError("DeepSeek PDF data URI must contain base64 PDF bytes")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("DeepSeek PDF data URI contains invalid base64") from None
    else:
        raise ValueError("DeepSeek does not support document URLs")
    if not isinstance(content, bytes) or not content:
        raise ValueError("DeepSeek PDF data must be non-empty bytes")
    if len(content) > MAX_PDF_BYTES:
        raise ValueError("DeepSeek PDFs must be at most 100 MiB")
    return content


def validate_raw_document_forms(messages: Any) -> Any:
    """Preflight and copy inline PDFs before Core normalization can decode them."""
    if isinstance(messages, Messages):
        raw_items: list[Any] = list(messages.items)
        as_messages = True
    elif isinstance(messages, (list, tuple)):
        raw_items = list(messages)
        as_messages = False
    else:
        return messages

    def prepare(part: Any, role: Any) -> Any:
        if isinstance(part, DocumentPart):
            if role != "user":
                raise ValueError("DeepSeek PDF parts are supported only in user messages")
            content = decode_pdf_part(part)
            if part.data is not None:
                return part
            return replace(
                part,
                url=None,
                data=content,
                media_type=part._get_media_type(),
            )
        if not isinstance(part, Mapping):
            return part
        part_type = part.get("type")
        source = part.get("source")
        is_document = part_type in {"document", "input_file", "file"}
        if isinstance(source, Mapping) and source.get("media_type") == "application/pdf":
            is_document = True
        if not is_document:
            return part
        if role != "user":
            raise ValueError("DeepSeek PDF parts are supported only in user messages")
        if part_type == "document" and isinstance(source, Mapping):
            if source.get("type") == "base64":
                raw = _strict_base64(source.get("data"))
                return DocumentPart(data=decode_pdf_part(DocumentPart(
                    data=raw,
                    media_type=source.get("media_type"),
                )), media_type=source.get("media_type"))
            if source.get("type") == "url":
                raise ValueError("DeepSeek does not support document URLs")
        if part_type == "input_file":
            if part.get("file_url"):
                raise ValueError("DeepSeek does not support document URLs")
            file_data = part.get("file_data")
            if file_data is not None:
                if not isinstance(file_data, str) or not file_data.startswith("data:"):
                    raise ValueError("DeepSeek does not support document URLs")
                return DocumentPart(
                    data=decode_pdf_part(DocumentPart(url=file_data)),
                    media_type="application/pdf",
                )
        if part_type == "file":
            file_obj = part.get("file")
            if isinstance(file_obj, Mapping):
                file_data = file_obj.get("file_data")
                if file_data is not None:
                    if not isinstance(file_data, str) or not file_data.startswith("data:"):
                        raise ValueError("DeepSeek does not support document URLs")
                    return DocumentPart(
                        data=decode_pdf_part(DocumentPart(url=file_data)),
                        media_type="application/pdf",
                    )
        return part

    for index, message in enumerate(raw_items):
        if isinstance(message, UserMessage):
            files = getattr(message, "files", None)
            if isinstance(files, list) and files:
                prepared_files = [prepare(part, message.role) for part in files]
                if any(prepared is not original for original, prepared in zip(files, prepared_files)):
                    raw_items[index] = replace(message, files=prepared_files)
            continue
        if not isinstance(message, Mapping):
            files = getattr(message, "files", None)
            if isinstance(files, list):
                for part in files:
                    prepare(part, getattr(message, "role", None))
            continue
        copied = dict(message)
        if isinstance(message.get("content"), list):
            copied["content"] = [prepare(part, message.get("role")) for part in message["content"]]
        if isinstance(message.get("files"), (list, tuple)):
            copied["files"] = [prepare(part, message.get("role")) for part in message["files"]]
        raw_items[index] = copied
    return Messages(raw_items) if as_messages else raw_items


def preserve_raw_pdf_parts(messages: Any) -> Any:
    """Move supported raw PDF blocks into Core's existing ``files`` input."""
    if not isinstance(messages, (list, tuple)):
        return messages
    result = list(messages)
    file_types = {"image_url", "input_image", "image", "document", "input_file", "file"}
    for index, message in enumerate(result):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        existing_files = message.get("files")
        raw_parts = [*content, *(existing_files or ())]
        if not any(
            isinstance(part, DocumentPart)
            or (
                isinstance(part, Mapping)
                and part.get("type") in {"document", "input_file", "file"}
            )
            for part in raw_parts
        ):
            continue
        text = " ".join(
            part["text"]
            for part in content
            if isinstance(part, Mapping)
            and part.get("type") in {"text", "input_text"}
            and isinstance(part.get("text"), str)
        )
        content_files = [
            part
            for part in content
            if isinstance(part, DocumentPart)
            or (isinstance(part, Mapping) and part.get("type") in file_types)
        ]
        normalizer = Messages([])
        files = [
            normalizer._normalize_dict_file_part(part)
            for part in [*content_files, *(existing_files or ())]
        ]
        result[index] = UserMessage(content=text, files=files or None)
    return result


def _strict_base64(value: Any) -> bytes:
    if not isinstance(value, str):
        raise ValueError("DeepSeek PDF base64 data must be a string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("DeepSeek PDF base64 data is invalid") from None


def pdf_slots(
    messages: Messages,
) -> tuple[PDFSlot, ...]:
    """Validate every normalized attachment and return PDFs in message order."""
    slots: list[PDFSlot] = []
    for message_index, message in enumerate(messages.items):
        files = getattr(message, "files", None)
        if files is None:
            continue
        if not isinstance(message, UserMessage):
            raise ValueError("DeepSeek file parts are supported only in user messages")
        if not isinstance(files, list):
            raise ValueError("DeepSeek user message files must be a list")
        for file_index, part in enumerate(files):
            if isinstance(part, ImagePart):
                continue
            if not isinstance(part, DocumentPart):
                raise ValueError("DeepSeek supports only image and PDF inputs")
            content = decode_pdf_part(part)
            slots.append(PDFSlot(message_index, file_index, content))
            if len(slots) > MAX_PDFS_PER_REQUEST:
                raise ValueError("DeepSeek supports at most 20 PDFs per request")
    return tuple(slots)
