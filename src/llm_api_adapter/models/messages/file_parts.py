from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


_URL_EXTENSION_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".pdf": "application/pdf",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}


@dataclass
class FilePart:
    url: Optional[str] = None
    data: Optional[bytes] = None
    media_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.url is None and self.data is None:
            raise ValueError("FilePart requires either url or data")
        if self.url is not None and self.data is not None:
            raise ValueError("FilePart accepts url or data, not both")
        if self.data is not None and self.media_type is None:
            raise ValueError("FilePart with data requires media_type")
        if self.url is not None and self.media_type is None:
            self.media_type = self._detect_from_url(self.url)

    def _is_url(self) -> bool:
        return self.url is not None and not self.url.startswith("data:")

    def _get_media_type(self) -> Optional[str]:
        if self.media_type:
            return self.media_type
        if self.url and self.url.startswith("data:"):
            m = re.match(r"data:([^;]+);", self.url)
            if m:
                return m.group(1)
        return None

    def _get_b64_data(self) -> str:
        if self.data is not None:
            return base64.b64encode(self.data).decode()
        return self.url.split(",", 1)[1]  # type: ignore[union-attr]

    def _to_data_uri(self) -> str:
        if self.url is not None and self.url.startswith("data:"):
            return self.url
        b64 = self._get_b64_data()
        return f"data:{self.media_type};base64,{b64}"

    @staticmethod
    def _detect_from_url(url: str) -> Optional[str]:
        suffix = Path(url.split("?")[0]).suffix.lower()
        return _URL_EXTENSION_MAP.get(suffix)


@dataclass
class ImagePart(FilePart):
    def __post_init__(self) -> None:
        super().__post_init__()
        mt = self._get_media_type()
        if mt is None:
            raise ValueError(
                "Cannot detect image media_type from URL — pass media_type explicitly"
            )
        if not mt.startswith("image/"):
            raise ValueError(
                f"ImagePart requires image/* media_type, got {mt!r}"
            )


# In 0.5.1 this will expand to Union[ImagePart, DocumentPart, AudioPart]
FileParts = Union[ImagePart]
