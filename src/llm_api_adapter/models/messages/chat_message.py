from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Tuple, Type

from ..tools import ToolCall
from .file_parts import DocumentPart, FilePart, ImagePart


@dataclass
class Message:
    content: str
    role: str = field(init=False)

    def to_openai(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}
    
    def to_openai_responses_input(self) -> List[Dict[str, Any]]:
        return [{"role": self.role, "content": self.content}]

    def to_anthropic(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

    def to_google(self) -> Dict[str, Any]:
        return {"role": self.role, "parts": [{"text": self.content}]}


@dataclass
class Prompt(Message):
    role: str = field(default="system", init=False)

    def to_anthropic(self) -> str:
        return self.content

    def to_google(self) -> str:
        return self.content


@dataclass
class UserMessage(Message):
    files: Optional[List[FilePart]] = None
    role: str = field(default="user", init=False)

    def to_openai(self) -> Dict[str, Any]:
        if self.files is None:
            return {"role": "user", "content": self.content}
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": self.content},
                *[self._part_to_openai_chat(p) for p in self.files],
            ],
        }

    def to_openai_responses_input(self) -> List[Dict[str, Any]]:
        if self.files is None:
            return [{"role": "user", "content": self.content}]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": self.content},
                    *[self._part_to_openai_responses(p) for p in self.files],
                ],
            }
        ]

    def to_anthropic(self) -> Dict[str, Any]:
        if self.files is None:
            return {"role": "user", "content": self.content}
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": self.content},
                *[self._part_to_anthropic(p) for p in self.files],
            ],
        }

    def to_google(self) -> Dict[str, Any]:
        if self.files is None:
            return {"role": "user", "parts": [{"text": self.content}]}
        return {
            "role": "user",
            "parts": [
                {"text": self.content},
                *[self._part_to_google(p) for p in self.files],
            ],
        }

    def _part_to_openai_chat(self, part: FilePart) -> Dict[str, Any]:
        if isinstance(part, ImagePart):
            url = part.url if part._is_url() else part._to_data_uri()
            return {"type": "image_url", "image_url": {"url": url}}
        if isinstance(part, DocumentPart):
            if part._is_url():
                raise ValueError(
                    "OpenAI Chat Completions does not support DocumentPart URL. "
                    "Use bytes/data, or switch to Responses API (gpt-5 models)."
                )
            return {
                "type": "file",
                "file": {
                    "filename": "document.pdf",
                    "file_data": part._to_data_uri(),
                },
            }
        raise ValueError(
            f"{type(part).__name__} is not supported as OpenAI Chat Completions file part"
        )

    def _part_to_openai_responses(self, part: FilePart) -> Dict[str, Any]:
        if isinstance(part, ImagePart):
            url = part.url if part._is_url() else part._to_data_uri()
            return {"type": "input_image", "image_url": url}
        if isinstance(part, DocumentPart):
            if part._is_url():
                return {"type": "input_file", "file_url": part.url}
            return {
                "type": "input_file",
                "filename": "document.pdf",
                "file_data": part._to_data_uri(),
            }
        raise ValueError(
            f"{type(part).__name__} is not supported as OpenAI Responses API file part"
        )

    def _part_to_anthropic(self, part: FilePart) -> Dict[str, Any]:
        if isinstance(part, ImagePart):
            if part._is_url():
                return {"type": "image", "source": {"type": "url", "url": part.url}}
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part._get_media_type(),
                    "data": part._get_b64_data(),
                },
            }
        if isinstance(part, DocumentPart):
            if part._is_url():
                return {
                    "type": "document",
                    "source": {"type": "url", "url": part.url},
                }
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": part._get_media_type(),
                    "data": part._get_b64_data(),
                },
            }
        raise ValueError(f"{type(part).__name__} is not supported as Anthropic file part")

    def _part_to_google(self, part: FilePart) -> Dict[str, Any]:
        if isinstance(part, ImagePart):
            if part._is_url():
                return {"fileData": {"mimeType": part._get_media_type(), "fileUri": part.url}}
            return {"inlineData": {"mimeType": part._get_media_type(), "data": part._get_b64_data()}}
        if isinstance(part, DocumentPart):
            if part._is_url():
                return {"fileData": {"mimeType": part._get_media_type(), "fileUri": part.url}}
            return {"inlineData": {"mimeType": part._get_media_type(), "data": part._get_b64_data()}}
        raise ValueError(f"{type(part).__name__} is not supported as Google file part")


@dataclass
class AIMessage(Message):
    """
    Assistant message.

    - OpenAI tool loop: may include tool_calls (assistant turn that requested tools).
    - Anthropic tool loop: tool_calls -> content blocks type="tool_use".
    - Google tool loop: tool_calls -> parts[].functionCall.
    """
    role: str = field(default="assistant", init=False)
    tool_calls: Optional[List[ToolCall]] = None

    def to_openai(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg
    
    def to_openai_responses_input(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if self.content:
            items.append({"role": "assistant", "content": self.content})
        return items

    def to_anthropic(self) -> Dict[str, Any]:
        if not self.tool_calls:
            return {"role": "assistant", "content": self.content}
        blocks: List[Dict[str, Any]] = []
        if self.content:
            blocks.append({"type": "text", "text": self.content})
        for tc in self.tool_calls:
            if not tc.call_id:
                raise ValueError(
                    "Anthropic tool_use requires a non-empty call_id (tool_use id)."
                )
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.call_id,
                    "name": tc.name,
                    "input": tc.arguments or {},
                }
            )
        return {"role": "assistant", "content": blocks}

    def to_google(self) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = []
        if self.content:
            parts.append({"text": self.content})
        if self.tool_calls:
            for tc in self.tool_calls:
                part: Dict[str, Any] = {
                    "functionCall": {
                        "name": tc.name,
                        "args": tc.arguments or {},
                    }
                }
                if tc.provider_data:
                    thought_signature = tc.provider_data.get("thoughtSignature")
                    if thought_signature is not None:
                        part["thoughtSignature"] = thought_signature
                parts.append(part)
        if not parts:
            parts = [{"text": ""}]
        return {"role": "model", "parts": parts}


@dataclass
class ToolMessage(Message):
    """
    Tool result message.

    - OpenAI format: role="tool" + tool_call_id
    - Anthropic format: user turn with tool_result block referencing tool_use_id
    - Google format: user turn with functionResponse part (name + response object)
      NOTE: Gemini doesn't have call_id; for Google tool loop we treat tool_call_id as function name.
    """
    tool_call_id: str
    role: str = field(default="tool", init=False)

    def to_openai(self) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}

    def to_openai_responses_input(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function_call_output",
                "call_id": self.tool_call_id,
                "output": self.content,
            }
        ]

    def to_anthropic(self) -> Dict[str, Any]:
        if not self.tool_call_id:
            raise ValueError("Anthropic tool_result requires tool_call_id (tool_use_id).")
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": self.tool_call_id,
                    "content": self.content,
                }
            ],
        }

    def to_google(self) -> Dict[str, Any]:
        name = self.tool_call_id
        if not name:
            raise ValueError("Google functionResponse requires tool_call_id to be set (use tool name).")
        response_obj: Any
        try:
            raw = self.content.strip()
            response_obj = json.loads(raw) if raw else {}
        except Exception:
            response_obj = {"content": self.content}
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": name,
                        "response": response_obj,
                    }
                }
            ],
        }


@dataclass
class Messages:
    items: List[Any]

    def __post_init__(self) -> None:
        role_to_cls = self._build_role_to_cls_map()
        self.items = [self._normalize_item(item, role_to_cls) for item in self.items]

    def _build_role_to_cls_map(self) -> Dict[str, Type[Message]]:
        return {cls.role: cls for cls in Message.__subclasses__()}

    def _normalize_item(
        self,
        item: Any,
        role_to_cls: Dict[str, Type[Message]],
    ) -> Message:
        if isinstance(item, Message):
            return item
        if isinstance(item, dict):
            return self._normalize_dict_item(item, role_to_cls)
        raise TypeError(f"Unsupported message type: {type(item)}")

    def _normalize_dict_item(
        self,
        item: Dict[str, Any],
        role_to_cls: Dict[str, Type[Message]],
    ) -> Message:
        role = item.get("role")
        if not role:
            raise ValueError("Missing 'role' in message data")
        if role not in role_to_cls:
            raise ValueError(f"Unsupported role: {role}")
        if role == "tool":
            return self._normalize_tool_message(item)
        if role == "assistant":
            return self._normalize_assistant_message(item)
        return self._normalize_simple_message(item, role_to_cls[role])

    def _normalize_tool_message(self, item: Dict[str, Any]) -> ToolMessage:
        tool_call_id = item.get("tool_call_id")
        if not tool_call_id:
            raise ValueError("Missing 'tool_call_id' for tool message")
        content = item.get("content")
        if content is None:
            raise ValueError("Missing 'content' in message data")
        return ToolMessage(content=str(content), tool_call_id=str(tool_call_id))

    def _normalize_assistant_message(self, item: Dict[str, Any]) -> AIMessage:
        content = item.get("content")
        if content is None:
            content = ""
        tool_calls_raw = item.get("tool_calls")
        tool_calls = self._parse_assistant_tool_calls(tool_calls_raw)
        return AIMessage(content=str(content), tool_calls=tool_calls)

    def _parse_assistant_tool_calls(
        self,
        tool_calls_raw: Any,
    ) -> Optional[List[ToolCall]]:
        if tool_calls_raw is None:
            return None
        if not isinstance(tool_calls_raw, list):
            raise ValueError("assistant.tool_calls must be a list")
        return [self._parse_single_tool_call(tc) for tc in tool_calls_raw]

    def _parse_single_tool_call(self, tool_call_raw: Any) -> ToolCall:
        if not isinstance(tool_call_raw, dict):
            raise ValueError("assistant.tool_calls items must be dict")
        if "name" in tool_call_raw:
            return self._parse_normalized_tool_call(tool_call_raw)
        return self._parse_legacy_openai_tool_call(tool_call_raw)

    def _parse_normalized_tool_call(self, tool_call_raw: Dict[str, Any]) -> ToolCall:
        name = tool_call_raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("assistant.tool_calls.name must be non-empty str")
        arguments = tool_call_raw.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError("assistant.tool_calls.arguments must be dict")
        call_id = tool_call_raw.get("call_id")
        provider_data = tool_call_raw.get("provider_data")
        if provider_data is not None and not isinstance(provider_data, dict):
            raise ValueError("assistant.tool_calls.provider_data must be dict")
        return ToolCall(
            name=name,
            arguments=arguments,
            call_id=call_id,
            provider_data=provider_data,
        )

    def _parse_legacy_openai_tool_call(self, tool_call_raw: Dict[str, Any]) -> ToolCall:
        function_payload = tool_call_raw.get("function") or {}
        name = function_payload.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "assistant.tool_calls.function.name must be non-empty str"
            )
        arguments = self._parse_legacy_openai_tool_arguments(
            function_payload.get("arguments", "{}")
        )
        call_id = tool_call_raw.get("id")
        return ToolCall(
            name=name,
            arguments=arguments,
            call_id=call_id,
        )

    def _parse_legacy_openai_tool_arguments(self, raw_args: Any) -> Dict[str, Any]:
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args) if raw_args.strip() else {}
            except Exception as e:
                raise ValueError(
                    f"assistant.tool_calls.arguments JSON parse failed: {e}"
                )
        if isinstance(raw_args, dict):
            return raw_args
        return {}

    def _normalize_simple_message(
        self,
        item: Dict[str, Any],
        message_cls: Type[Message],
    ) -> Message:
        content = item.get("content")
        if message_cls is UserMessage:
            if isinstance(content, list):
                text = " ".join(p["text"] for p in content if p.get("type") == "text")
                files = [
                    self._normalize_dict_file_part(p) for p in content
                    if p.get("type") in ("image_url", "input_image", "image")
                ]
                return UserMessage(content=text, files=files or None)
            if not content:
                raise ValueError("Missing 'content' in message data")
            files_raw = item.get("files")
            files = [self._normalize_dict_file_part(p) for p in files_raw] if files_raw else None
            return UserMessage(content=str(content), files=files)
        if content is None or content == "":
            raise ValueError("Missing 'content' in message data")
        return message_cls(content=str(content))

    def _normalize_dict_file_part(self, part: Any) -> FilePart:
        if isinstance(part, FilePart):
            return part
        if not isinstance(part, dict):
            raise ValueError(f"Unsupported file part type: {type(part)}")
        part_type = part.get("type")
        if part_type == "image_url":
            return self._normalize_image_url_part(part)
        if part_type == "input_image":
            return self._normalize_openai_input_image_part(part)
        if part_type == "image":
            return self._normalize_anthropic_image_part(part)
        if part_type == "document":
            return self._normalize_anthropic_document_part(part)
        if part_type == "input_file":
            return self._normalize_openai_input_file_part(part)
        if part_type == "file":
            return self._normalize_openai_chat_file_part(part)
        raise ValueError(f"Unsupported file part format: {part_type!r}")

    def _normalize_image_url_part(self, part: Dict[str, Any]) -> ImagePart:
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            return ImagePart(url=image_url["url"])
        return ImagePart(url=image_url)

    def _normalize_openai_input_image_part(self, part: Dict[str, Any]) -> ImagePart:
        return ImagePart(url=part["image_url"])

    def _normalize_anthropic_image_part(self, part: Dict[str, Any]) -> ImagePart:
        source = part.get("source", {})
        if source.get("type") == "url":
            return ImagePart(url=source["url"])
        if source.get("type") == "base64":
            return ImagePart(
                data=base64.b64decode(source["data"]),
                media_type=source["media_type"],
            )
        raise ValueError(f"Unsupported file part format: {part.get('type')!r}")

    def _normalize_anthropic_document_part(self, part: Dict[str, Any]) -> DocumentPart:
        source = part.get("source", {})
        if source.get("type") == "url":
            return DocumentPart(url=source["url"])
        if source.get("type") == "base64":
            return DocumentPart(
                data=base64.b64decode(source["data"]),
                media_type=source["media_type"],
            )
        raise ValueError(f"Unsupported file part format: {part.get('type')!r}")

    def _normalize_openai_input_file_part(self, part: Dict[str, Any]) -> DocumentPart:
        file_url = part.get("file_url")
        if file_url:
            return DocumentPart(url=file_url)
        file_data = part.get("file_data")
        if file_data:
            return DocumentPart(url=file_data)
        raise ValueError(f"Unsupported file part format: {part.get('type')!r}")

    def _normalize_openai_chat_file_part(self, part: Dict[str, Any]) -> DocumentPart:
        file_obj = part.get("file", {})
        file_data = file_obj.get("file_data")
        if file_data:
            return DocumentPart(url=file_data)
        raise ValueError(f"Unsupported file part format: {part.get('type')!r}")
   
    def to_openai(self) -> List[Dict[str, Any]]:
        return [m.to_openai() for m in self.items]
    
    def to_openai_responses_input(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for m in self.items:
            if isinstance(m, Prompt):
                continue
            items.extend(m.to_openai_responses_input())
        return items

    def to_openai_responses_instructions(self) -> Optional[str]:
        instructions: Optional[str] = None
        for m in self.items:
            if isinstance(m, Prompt):
                if instructions is not None:
                    raise ValueError(
                        "Multiple system prompts are not allowed for OpenAI Responses."
                    )
                instructions = m.content
        return instructions

    def to_anthropic(self) -> Tuple[str | None, List[Dict[str, Any]]]:
        prompt: Optional[str] = None
        messages: List[Dict[str, Any]] = []
        for m in self.items:
            if isinstance(m, Prompt):
                if prompt is not None:
                    raise ValueError("Multiple system prompts are not allowed for Anthropic.")
                prompt = m.to_anthropic()
                continue
            if isinstance(m, ToolMessage):
                messages.append(m.to_anthropic())
                continue
            if isinstance(m, AIMessage):
                messages.append(m.to_anthropic())
                continue
            messages.append(m.to_anthropic())
        return prompt, messages

    def to_google(self) -> Tuple[str | None, List[Dict[str, Any]]]:
        prompt: Optional[str] = None
        messages: List[Dict[str, Any]] = []
        for m in self.items:
            if isinstance(m, Prompt):
                if prompt is not None:
                    raise ValueError("Multiple system prompts are not allowed for Google.")
                prompt = m.to_google()
                continue
            if isinstance(m, ToolMessage):
                messages.append(m.to_google())
                continue
            if isinstance(m, AIMessage):
                messages.append(m.to_google())
                continue
            if isinstance(m, UserMessage):
                messages.append(m.to_google())
            else:
                messages.append({"role": "user", "parts": [{"text": m.content}]})
        return prompt, messages
