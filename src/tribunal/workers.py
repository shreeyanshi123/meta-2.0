"""Simulated worker agents for the Tribunal environment.

Workers are deterministic template-based agents (not trained). They produce
realistic-looking outputs from TaskBriefs. An optional LLM fallback is
available behind the ``use_llm`` flag.
"""

from __future__ import annotations


import random
import re
import textwrap
from typing import List, Optional, Protocol

from tribunal.schemas import TaskBrief, WorkerOutput, WorkerRole


# ---------------------------------------------------------------------------
# LLM Backend abstraction
# ---------------------------------------------------------------------------

class LLMBackend(Protocol):
    """Thin adapter interface for an optional local LLM."""

    def generate(self, system: str, user: str) -> str: ...  # pragma: no cover


class NullLLMBackend:
    """Default backend that raises if called — forces template path."""

    def generate(self, system: str, user: str) -> str:
        raise RuntimeError(
            "NullLLMBackend.generate() called. Set use_llm=False or provide a real backend."
        )


class UnslothBackend:
    """Stub for Unsloth-loaded Qwen2.5-0.5B. Import-guarded; does NOT load weights."""

    def __init__(self, model_name: str = "unsloth/Qwen2.5-0.5B-Instruct") -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from unsloth import FastLanguageModel  # type: ignore[import-untyped]

            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=1024,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self._model)
        except ImportError as exc:
            raise ImportError(
                "unsloth is not installed. Install it or use use_llm=False."
            ) from exc

    def generate(self, system: str, user: str) -> str:
        self._lazy_load()
        assert self._tokenizer is not None and self._model is not None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        output_ids = self._model.generate(
            input_ids=input_ids, max_new_tokens=512, temperature=0.7
        )
        return self._tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\$?[\d,]+(?:\.\d+)?%?")


def _extract_numbers(text: str) -> list[str]:
    """Return all numeric-looking tokens from *text*."""
    return _NUM_RE.findall(text)


# ---------------------------------------------------------------------------
# Per-role clean producers (template-based, deterministic)
# ---------------------------------------------------------------------------

def _produce_summariser(brief: TaskBrief, rng: random.Random) -> str:
    """Produce a 3-bullet summary citing numeric facts from source_material."""
    source = brief.source_material or ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source) if s.strip()]

    # Pick 3 sentences that contain numbers; fall back to first 3
    numeric_sentences = [s for s in sentences if _NUM_RE.search(s)]
    if len(numeric_sentences) < 3:
        numeric_sentences = sentences[:3]
    chosen = numeric_sentences[:3]
    rng.shuffle(chosen)

    bullets = []
    for i, sent in enumerate(chosen):
        # Light rewrite: trim to one sentence, keep numbers intact
        bullets.append(f"• {sent}")

    return "\n".join(bullets) if bullets else "• No data available."


def _produce_ticket_resolver(brief: TaskBrief, rng: random.Random) -> str:
    """Produce resolution steps referencing the KB article."""
    kb = brief.private_context.get("kb_article", "")
    source = brief.source_material or ""

    # Extract the ticket ID
    ticket_match = re.search(r"Ticket\s*#(\d+)", source)
    ticket_id = ticket_match.group(0) if ticket_match else "the reported ticket"

    # Parse numbered steps from the KB article
    steps = re.findall(r"\d\)\s*([^.]+\.?[^0-9)]*)", kb)
    if not steps:
        steps = ["Investigate the reported issue.", "Apply the recommended fix.", "Verify resolution."]

    lines = [f"Resolution for {ticket_id}:\n"]
    for i, step in enumerate(steps[:5], 1):
        lines.append(f"Step {i}: {step.strip()}")
    lines.append("\nPlease verify the fix and confirm resolution.")
    return "\n".join(lines)


def _produce_negotiator(brief: TaskBrief, rng: random.Random) -> str:
    """Propose a counter-offer within a sensible band."""
    budget_raw = brief.private_context.get("budget_ceiling", "$0")
    walk_raw = brief.private_context.get("walk_away_price", "$0")
    counter_raw = brief.private_context.get("counter_offer_from_supplier", "")



    source = brief.source_material or ""

    proposal = textwrap.dedent(f"""\
        After reviewing the supplier's proposal, I recommend the following counter-offer:

        Current offer summary: {source[:200]}...

        Our counter-proposal:
        - We propose a price point aligned with our budget ceiling of {budget_raw}.
        - The supplier has indicated willingness to offer: {counter_raw}.
        - Our walk-away threshold is {walk_raw}.

        Justification:
        Given current market rates and our projected demand, the supplier's initial
        quote is above our target range. The counter-offer represents a fair
        compromise that keeps us within budget while maintaining the relationship.

        Recommendation: Accept if the supplier meets or comes within 5% of our
        walk-away price. Otherwise, evaluate alternative providers.
    """)
    return proposal.strip()


def _produce_researcher(brief: TaskBrief, rng: random.Random) -> str:
    """List exactly 3 citations from the provided valid sources."""
    valid_sources = brief.private_context.get("valid_sources", [])
    if isinstance(valid_sources, list) and len(valid_sources) >= 3:
        chosen = rng.sample(valid_sources, 3)
    else:
        chosen = list(valid_sources)[:3] if valid_sources else [
            "Source unavailable [1]",
            "Source unavailable [2]",
            "Source unavailable [3]",
        ]

    claim = brief.source_material or "the given claim"
    lines = [f"Citations supporting {claim[:120]}:\n"]
    for i, src in enumerate(chosen, 1):
        lines.append(f"[{i}] {src}")
    return "\n".join(lines)


_PRODUCERS = {
    WorkerRole.SUMMARISER: _produce_summariser,
    WorkerRole.TICKET_RESOLVER: _produce_ticket_resolver,
    WorkerRole.NEGOTIATOR: _produce_negotiator,
    WorkerRole.RESEARCHER: _produce_researcher,
}


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------

class WorkerAgent:
    """A single simulated worker agent.

    Parameters
    ----------
    role : WorkerRole
        Which role this worker fulfils.
    seed : int
        Base seed for deterministic output.
    use_llm : bool
        If True, use *llm_backend* instead of templates.
    llm_backend : LLMBackend | None
        Backend to call when *use_llm* is True. Defaults to :class:`NullLLMBackend`.
    """

    def __init__(
        self,
        role: WorkerRole,
        seed: int,
        use_llm: bool = False,
        llm_backend: Optional[LLMBackend] = None,
    ) -> None:
        self.role = role
        self.seed = seed
        self.use_llm = use_llm
        self.llm_backend: LLMBackend = llm_backend or NullLLMBackend()

    def run(self, brief: TaskBrief) -> WorkerOutput:
        """Produce a :class:`WorkerOutput` for the given :class:`TaskBrief`."""
        rng = random.Random(self.seed + brief.worker_id)

        if self.use_llm:
            content = self._run_llm(brief)
        else:
            producer = _PRODUCERS[self.role]
            content = producer(brief, rng)

        tokens_used = len(content.split())
        return WorkerOutput(
            worker_id=brief.worker_id,
            role=self.role,
            content=content,
            self_confidence=round(rng.uniform(0.5, 0.95), 2),
            tokens_used=tokens_used,
        )

    def _run_llm(self, brief: TaskBrief) -> str:
        system_prompt = (
            f"You are a {self.role.value} agent. Complete the task accurately and concisely."
        )
        user_prompt = (
            f"Task: {brief.prompt}\n\n"
            f"Source material:\n{brief.source_material or 'N/A'}\n\n"
            f"Context:\n{brief.private_context}"
        )
        return self.llm_backend.generate(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# WorkerPool
# ---------------------------------------------------------------------------

_ROLE_TO_WORKER_ID = {
    WorkerRole.SUMMARISER: 0,
    WorkerRole.TICKET_RESOLVER: 1,
    WorkerRole.NEGOTIATOR: 2,
    WorkerRole.RESEARCHER: 3,
}


class WorkerPool:
    """Runs all 4 workers for a round (synchronous for reproducibility).

    Parameters
    ----------
    seed : int
        Base seed forwarded to each :class:`WorkerAgent`.
    use_llm : bool
        Whether to use LLM backends for all workers.
    llm_backend : LLMBackend | None
        Shared backend instance (if *use_llm* is True).
    """

    def __init__(
        self,
        seed: int,
        use_llm: bool = False,
        llm_backend: Optional[LLMBackend] = None,
    ) -> None:
        self.agents: dict[WorkerRole, WorkerAgent] = {
            role: WorkerAgent(role=role, seed=seed, use_llm=use_llm, llm_backend=llm_backend)
            for role in WorkerRole
        }

    def run_all(self, briefs: List[TaskBrief]) -> List[WorkerOutput]:
        """Run each worker on its matching brief. Returns outputs in worker_id order."""
        brief_map = {b.role: b for b in briefs}
        outputs: list[WorkerOutput] = []
        for role in WorkerRole:
            agent = self.agents[role]
            brief = brief_map[role]
            outputs.append(agent.run(brief))
        return outputs
