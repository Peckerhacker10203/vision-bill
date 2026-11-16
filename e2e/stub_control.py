"""Test-owned source/photo plans and observable barriers shared by CPU stubs."""

import copy
import threading


class Scenario:
    def __init__(self):
        self.lock = threading.RLock()
        self.requests = []
        self.plans = {}
        self.barriers = {}

    def configure(self, body):
        plans = {}
        for plan in body.get("plans", []):
            key = (plan["source_sha256"], plan["photo_index"])
            if key in plans or plan["result"] not in ("success", "empty", "503", "timeout"):
                raise ValueError("invalid or duplicate source/photo plan")
            plans[key] = dict(plan)
        with self.lock:
            for event in self.barriers.values():
                event.set()
            self.plans = plans
            self.requests.clear()
            self.barriers = {name: threading.Event() for name in body.get("barriers", [])}

    def plan(self, digest, index):
        with self.lock:
            return copy.deepcopy(self.plans.get((digest, index), {"result": "success"}))

    def record(self, entry):
        with self.lock:
            self.requests.append(entry)

    def wait(self, name, timeout):
        with self.lock:
            event = self.barriers[name]
        return event.wait(timeout)

    def release(self, name):
        with self.lock:
            self.barriers[name].set()

    def audit(self):
        with self.lock:
            return copy.deepcopy(self.requests)
