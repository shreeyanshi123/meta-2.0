"""Tribunal Client — sync and async HTTP clients for the Tribunal environment.

Usage::

    from tribunal_client import TribunalClient

    client = TribunalClient("http://localhost:8000")
    obs = client.reset()
    result = client.step(verdict)
    print(result.reward)
"""

from tribunal_client.client import TribunalClient, AsyncTribunalClient

__all__ = ["TribunalClient", "AsyncTribunalClient"]
