"""Sync and async HTTP clients for the AI Agent Oversight Tribunal.

IMPORT BOUNDARY: This module imports ONLY from ``tribunal_shared``.
It has zero dependencies on any server-side module (env, server,
failure_injector, rewards).  This is enforced by test.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

import httpx

from tribunal_shared.schemas import (
    JudgeVerdict,
    Observation,
    State,
    StepResult,
)


class TribunalClient:
    """Synchronous HTTP client for the Tribunal environment.

    Parameters
    ----------
    base_url : str
        Base URL of the Tribunal server (e.g. ``http://localhost:8000``).
    timeout : float
        Request timeout in seconds.

    Example
    -------
    >>> client = TribunalClient("http://localhost:8000")
    >>> obs = client.reset()
    >>> result = client.step(verdict)
    >>> print(result.reward, result.done)
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=timeout)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "TribunalClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment and return the first Observation.

        Parameters
        ----------
        seed : int | None
            Optional seed override.
        """
        body = {}
        if seed is not None:
            body["seed"] = seed
        r = self._http.post("/reset", json=body if body else None)
        r.raise_for_status()
        return Observation.model_validate(r.json())

    def step(self, verdict: JudgeVerdict) -> StepResult:
        """Submit a JudgeVerdict and return the step result.

        Parameters
        ----------
        verdict : JudgeVerdict
            The Judge's structured output.

        Returns
        -------
        StepResult
            Contains observation, reward, done, and info.
        """
        payload = {"verdict": verdict.model_dump()}
        r = self._http.post("/step", json=payload)
        r.raise_for_status()
        data = r.json()

        # Parse the observation within the response
        obs = Observation.model_validate(data["observation"])
        return StepResult(
            observation=obs,
            reward=data["reward"],
            done=data["done"],
            info=data["info"],
        )

    def state(self) -> State:
        """Return the current environment state."""
        r = self._http.get("/state")
        r.raise_for_status()
        return State.model_validate(r.json())

    def health(self) -> Dict[str, Any]:
        """Health check."""
        r = self._http.get("/health")
        r.raise_for_status()
        return r.json()

    def info(self) -> Dict[str, Any]:
        """Get environment metadata."""
        r = self._http.get("/info")
        r.raise_for_status()
        return r.json()

    def stream_rounds(self) -> Iterator[Dict[str, Any]]:
        """Consume /stream/rounds Server-Sent Events.

        Yields
        ------
        dict
            Each completed round's data (task_briefs, worker_outputs,
            ground_truth, verdict, reward_breakdown).
        """
        with self._http.stream("GET", "/stream/rounds") as response:
            response.raise_for_status()
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.split("\n"):
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                pass


class AsyncTribunalClient:
    """Asynchronous HTTP client for the Tribunal environment.

    Parameters
    ----------
    base_url : str
        Base URL of the Tribunal server.
    timeout : float
        Request timeout in seconds.

    Example
    -------
    >>> async with AsyncTribunalClient("http://localhost:8000") as client:
    ...     obs = await client.reset()
    ...     result = await client.step(verdict)
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncTribunalClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment and return the first Observation."""
        body = {}
        if seed is not None:
            body["seed"] = seed
        r = await self._http.post("/reset", json=body if body else None)
        r.raise_for_status()
        return Observation.model_validate(r.json())

    async def step(self, verdict: JudgeVerdict) -> StepResult:
        """Submit a JudgeVerdict and return the step result."""
        payload = {"verdict": verdict.model_dump()}
        r = await self._http.post("/step", json=payload)
        r.raise_for_status()
        data = r.json()
        obs = Observation.model_validate(data["observation"])
        return StepResult(
            observation=obs,
            reward=data["reward"],
            done=data["done"],
            info=data["info"],
        )

    async def state(self) -> State:
        """Return the current environment state."""
        r = await self._http.get("/state")
        r.raise_for_status()
        return State.model_validate(r.json())

    async def health(self) -> Dict[str, Any]:
        """Health check."""
        r = await self._http.get("/health")
        r.raise_for_status()
        return r.json()

    async def info(self) -> Dict[str, Any]:
        """Get environment metadata."""
        r = await self._http.get("/info")
        r.raise_for_status()
        return r.json()

    async def stream_rounds(self):
        """Consume /stream/rounds Server-Sent Events.

        Yields dicts of completed round data.
        """
        async with self._http.stream("GET", "/stream/rounds") as response:
            response.raise_for_status()
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.split("\n"):
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                pass
