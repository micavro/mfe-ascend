"""单次请求：id、prompt、template、op_output、benchmark 等。"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional


class Query:
    def __init__(
        self,
        id: Optional[str],
        prompt: str,
        priority: int = 0,
        template: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = id if id is not None else str(uuid.uuid4())
        self.prompt = prompt
        self.template = template or ""
        self.prompt_len = len(prompt) if prompt else 0
        self.metadata: Dict[str, Any] = dict(metadata or {})
        raw_est = self.metadata.get("input_len_est_tokens")
        self.input_len_est_tokens = int(raw_est) if raw_est not in (None, "") else (
            max(1, (self.prompt_len + 3) // 4) if self.prompt_len else 0
        )
        self.status = "pending"
        self.error_type = None
        self.error_message = None
        self.failed_op = None
        self.worker_id = None
        self.priority = priority
        self.op_output: Dict[str, str] = {}
        self.step = 0
        self.create_time = time.perf_counter()
        self.benchmark: Dict[str, Any] = {}
        self.op_metrics: Dict[str, Dict[str, Any]] = {}
        self.worker_assignments: Dict[str, int] = {}

        # Filled by MultiRequestOptimizer when MFE_SCHEDULER=sailp.
        self.schedule_plan = None
        self.scheduler_name = "eager"
