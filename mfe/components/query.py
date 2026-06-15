"""单次请求：id、prompt、template、op_output、benchmark 等。"""

import uuid
import time


class Query:
    def __init__(self, id, prompt, priority=0, template=""):
        self.id = id if id is not None else uuid.uuid4()
        self.prompt = prompt
        self.template = template or ""
        self.prompt_len = len(prompt) if prompt else 0
        self.status = "pending"
        self.error_type = None
        self.error_message = None
        self.failed_op = None
        self.worker_id = None
        self.priority = priority
        self.op_output = {}
        self.step = 0
        self.create_time = time.perf_counter()
        self.benchmark = {}
        self.worker_assignments = {}
