from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Hazard:
    camera_id: str
    camera_name: str
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    recommendation: str
    alert: bool
    timestamp: float
    source: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data
