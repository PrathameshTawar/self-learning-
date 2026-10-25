"""
Job dispatch queues. A delivery is acknowledged only after its result is stored, so a worker that
dies mid-job leaves the job pending; once its heartbeat stops, another worker reclaims it.
"""

from abc import ABC, abstractmethod
import threading
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Delivery(BaseModel):
    message_id: str
    job_id: str
    deliveries: int


class IJobQueue(ABC):
    @abstractmethod
    def enqueue(self, job_id: str) -> None: ...

    @abstractmethod
    def reserve(self, consumer: str, block_seconds: float = 5.0) -> Optional[Delivery]:
        """Next job for `consumer`: a stale delivery from a dead worker first, otherwise a new one."""

    @abstractmethod
    def ack(self, delivery: Delivery) -> None: ...

    @abstractmethod
    def heartbeat(self, delivery: Delivery, consumer: str) -> bool:
        """Keep a delivery from being reclaimed; False if another worker has already reclaimed it."""

    @abstractmethod
    def depth(self) -> Dict[str, int]: ...

    def ping(self) -> bool:
        return True


class InMemoryJobQueue(IJobQueue):
    """Thread-safe process-local queue with the same reclaim semantics, for tests and single-process mode."""

    def __init__(self, visibility_timeout_seconds: float = 300.0) -> None:
        self.visibility_timeout = visibility_timeout_seconds
        self._ready: List[str] = []
        self._in_flight: Dict[str, Dict[str, Any]] = {}
        self._deliveries: Dict[str, int] = {}
        self._counter = 0
        self._cond = threading.Condition()

    def enqueue(self, job_id: str) -> None:
        with self._cond:
            self._counter += 1
            message_id = f"{self._counter}-0"
            self._ready.append(message_id)
            self._in_flight[message_id] = {"job_id": job_id, "consumer": None, "seen": 0.0}
            self._cond.notify()

    def _deliver(self, message_id: str, consumer: str) -> Delivery:
        entry = self._in_flight[message_id]
        entry["consumer"], entry["seen"] = consumer, time.monotonic()
        self._deliveries[message_id] = self._deliveries.get(message_id, 0) + 1
        return Delivery(message_id=message_id, job_id=entry["job_id"], deliveries=self._deliveries[message_id])

    def reserve(self, consumer: str, block_seconds: float = 5.0) -> Optional[Delivery]:
        deadline = time.monotonic() + block_seconds
        with self._cond:
            while True:
                now = time.monotonic()
                for message_id, entry in self._in_flight.items():
                    if entry["consumer"] is not None and now - entry["seen"] >= self.visibility_timeout:
                        return self._deliver(message_id, consumer)
                if self._ready:
                    return self._deliver(self._ready.pop(0), consumer)
                remaining = deadline - now
                if remaining <= 0:
                    return None
                self._cond.wait(min(remaining, 0.5))

    def ack(self, delivery: Delivery) -> None:
        with self._cond:
            self._in_flight.pop(delivery.message_id, None)
            self._deliveries.pop(delivery.message_id, None)

    def heartbeat(self, delivery: Delivery, consumer: str) -> bool:
        with self._cond:
            entry = self._in_flight.get(delivery.message_id)
            if entry is None or entry["consumer"] != consumer:
                return False
            entry["seen"] = time.monotonic()
            return True

    def depth(self) -> Dict[str, int]:
        with self._cond:
            return {"waiting": len(self._ready), "in_flight": len(self._in_flight) - len(self._ready)}


# redis-py 8 defaults to a 5 s socket timeout, which a blocking XREADGROUP would trip; blocks are
# capped below an explicit timeout instead
REDIS_SOCKET_TIMEOUT_SECONDS = 30.0
MAX_BLOCK_SECONDS = 20.0

# Refresh the idle time only while this consumer still owns the delivery. Checking and claiming in
# one script keeps a slow worker from taking back a job another worker already reclaimed.
HEARTBEAT_SCRIPT = """
local owned = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[2], ARGV[2], 1, ARGV[3])
if #owned == 0 then return 0 end
redis.call('XCLAIM', KEYS[1], ARGV[1], ARGV[3], 0, ARGV[2], 'JUSTID')
return 1
"""


class RedisStreamQueue(IJobQueue):
    """
    Redis Streams with one consumer group. Unacknowledged deliveries stay in the group's pending
    list; XAUTOCLAIM hands those idle longer than the visibility timeout to a live worker, and the
    pending list's delivery counter tells the worker how many times a job has been tried.
    """

    def __init__(
        self,
        url: str,
        stream: str = "evoagent:jobs",
        group: str = "workers",
        visibility_timeout_seconds: int = 300,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise RuntimeError('Redis support needs: pip install -e ".[server]"') from exc
            client = redis.Redis.from_url(
                url, decode_responses=True, socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS, socket_connect_timeout=10,
                health_check_interval=30,
            )
        self.client = client
        self.stream = stream
        self.group = group
        self.visibility_ms = int(visibility_timeout_seconds * 1000)
        self._heartbeat = self.client.register_script(HEARTBEAT_SCRIPT)
        try:
            self.client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def enqueue(self, job_id: str) -> None:
        self.client.xadd(self.stream, {"job_id": job_id})

    def _times_delivered(self, message_id: str) -> int:
        entries = self.client.xpending_range(self.stream, self.group, min=message_id, max=message_id, count=1)
        return int(entries[0]["times_delivered"]) if entries else 1

    def reserve(self, consumer: str, block_seconds: float = 5.0) -> Optional[Delivery]:
        claimed = self.client.xautoclaim(
            self.stream, self.group, consumer, min_idle_time=self.visibility_ms, start_id="0-0", count=1
        )
        messages = claimed[1] if claimed else []
        if not messages:
            response = self.client.xreadgroup(
                self.group, consumer, {self.stream: ">"}, count=1,
                block=max(1, int(min(block_seconds, MAX_BLOCK_SECONDS) * 1000)),
            )
            messages = response[0][1] if response else []
        for message_id, fields in messages:
            if not fields:
                # Trimmed from the stream while pending: nothing left to run
                self.client.xack(self.stream, self.group, message_id)
                continue
            return Delivery(message_id=message_id, job_id=fields["job_id"], deliveries=self._times_delivered(message_id))
        return None

    def ack(self, delivery: Delivery) -> None:
        pipe = self.client.pipeline()
        pipe.xack(self.stream, self.group, delivery.message_id)
        pipe.xdel(self.stream, delivery.message_id)
        pipe.execute()

    def heartbeat(self, delivery: Delivery, consumer: str) -> bool:
        return bool(self._heartbeat(keys=[self.stream], args=[self.group, delivery.message_id, consumer]))

    def depth(self) -> Dict[str, int]:
        pending = self.client.xpending(self.stream, self.group)["pending"]
        return {"waiting": max(0, self.client.xlen(self.stream) - pending), "in_flight": pending}

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False
