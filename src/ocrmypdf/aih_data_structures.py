# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DetectorSource(str, Enum):
    CHROME_SCREENAI = "chrome_screenai"
    PADDLE_DET = "paddle_det"
    TESSERACT = "tesseract"
    PROJECTION_DEBUG = "projection_debug"


class ContentType(int, Enum):
    PRINTED_TEXT = 0
    HANDWRITTEN_TEXT = 1
    IMAGE = 2
    LINE_DRAWING = 3
    SEPARATOR = 4
    UNREADABLE_TEXT = 5
    FORMULA = 6
    HANDWRITTEN_FORMULA = 7
    SIGNATURE = 8
    UNKNOWN = -1

    @classmethod
    def from_value(cls, value: int | None) -> "ContentType":
        if value is None:
            return cls.UNKNOWN
        try:
            return cls(int(value))
        except ValueError:
            return cls.UNKNOWN


class TextDirection(int, Enum):
    UNSPECIFIED = 0
    LEFT_TO_RIGHT = 1
    RIGHT_TO_LEFT = 2
    TOP_TO_BOTTOM = 3

    @classmethod
    def from_value(cls, value: int | None) -> "TextDirection":
        if value is None:
            return cls.UNSPECIFIED
        try:
            return cls(int(value))
        except ValueError:
            return cls.UNSPECIFIED


@dataclass(slots=True)
class LineBox:
    page: int
    line: int
    bbox: tuple[int, int, int, int]
    poly: list[tuple[int, int]]
    confidence: float = 0.0
    source: str = DetectorSource.PROJECTION_DEBUG.value
    text: str = ""
    content_type: ContentType = ContentType.UNKNOWN
    direction: TextDirection = TextDirection.UNSPECIFIED
    block_id: int | None = None
    paragraph_id: int | None = None
    language: str | None = None
    words: list[tuple[str, tuple[int, int, int, int] | None]] = field(default_factory=list)


def bbox_from_xywh(x: float, y: float, width: float, height: float) -> tuple[int, int, int, int]:
    x0 = int(round(x))
    y0 = int(round(y))
    x1 = int(round(x + width))
    y1 = int(round(y + height))
    return (x0, y0, x1, y1)


def poly_from_bbox(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
