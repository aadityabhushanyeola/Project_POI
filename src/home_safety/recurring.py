from __future__ import annotations

import json
import threading
from pathlib import Path

from src.home_safety.recommender import recommend
from src.home_safety.types import Hazard


class RecurringZoneTracker:
    def __init__(self, path: str, alert_count: int, cell_size: int = 80):
        self.path = Path(path)
        self.alert_count = alert_count
        self.cell_size = cell_size
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {}
        if self.path.exists():
            self.counts = json.loads(self.path.read_text(encoding="utf-8"))

    def update(self, hazard: Hazard) -> Hazard | None:
        x1, y1, x2, y2 = hazard.bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        key = f"{hazard.camera_id}:{cx // self.cell_size}:{cy // self.cell_size}:{hazard.label}"
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1
            count = self.counts[key]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.counts, indent=2), encoding="utf-8")

        if count == self.alert_count:
            return Hazard(
                camera_id=hazard.camera_id,
                camera_name=hazard.camera_name,
                label="recurring_hazard_zone",
                confidence=0.8,
                bbox=hazard.bbox,
                recommendation=recommend("recurring_hazard_zone", True),
                alert=True,
                timestamp=hazard.timestamp,
                source="recurring_counter",
            )
        return None
