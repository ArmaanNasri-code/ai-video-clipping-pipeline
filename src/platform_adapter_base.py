"""
Base interface for streaming platform adapters.

The idea: everything downstream of chat/VOD fetching (spike detection,
LLM scoring, merging, clip cropping/captioning) doesn't care which
platform a stream came from. Each platform (Kick, Twitch, etc.) gets
its own adapter implementing these four methods; the rest of the
pipeline just calls the adapter and doesn't need to know which
platform it's talking to.
"""

from abc import ABC, abstractmethod
from datetime import datetime


class StreamPlatformAdapter(ABC):
    @abstractmethod
    def get_stream_start_time(self, stream_id: str) -> datetime:
        """Returns the stream's start timestamp, used for elapsed-time math
        (stream_start + elapsed = real wall-clock timestamp for VOD lookup)."""
        raise NotImplementedError

    @abstractmethod
    def get_vod_manifest(self, stream_id: str) -> str:
        """Returns a local file path to a fresh, valid HLS manifest for
        this stream's VOD."""
        raise NotImplementedError

    @abstractmethod
    def get_chat_log(self, stream_id: str, start_time: datetime, end_time: datetime) -> list:
        """Returns chat messages as a list of (timestamp, username, message)
        tuples, in the same format your spike-detection logic already expects."""
        raise NotImplementedError

    @abstractmethod
    def fetch_segment_window(self, manifest_path: str, target_time: datetime, window_seconds: int) -> str:
        """Downloads and concatenates HLS segments around a target timestamp.
        Returns the local output video file path."""
        raise NotImplementedError
