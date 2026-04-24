# Training an AI Judge to Catch Misbehaving Agents

*An open-source environment and GRPO training pipeline for multi-agent oversight*

---

As AI agents are deployed in production pipelines — summarising documents, resolving support tickets, negotiating contracts — a critical question emerges: **who watches the watchers?** A single hallucinating agent can cascade errors through an entire workflow, and current approaches to oversight (human review, simple heuristics) either don't scale or are trivially gamed.

We built the **AI Agent Oversight Tribunal**, an OpenEnv-compatible reinforcement learning environment where a Judge LLM learns to detect, classify, and explain failures injected into a team of AI workers.

## The Setup

Four specialised workers — a Summariser, Ticket Resolver, Negotiator, and Researcher — each receive task briefs with private context and produce structured outputs. A **hidden failure injector** probabilistically corrupts their work using one of four strategies:

- **Hallucination**: Fabricated statistics, fake DOI citations, invented metrics
- **Collusion**: Leaking distinctive tokens from another worker's confidential context
- **Manipulation**: Biased framing with loaded adjectives and misleading conclusions
- **Silence**: Dropping required output components (missing citations, truncated steps)

The Judge receives only the (potentially corrupted) outputs — never the ground truth — and must issue a structured verdict: which workers misbehaved, what kind of failure occurred, and a reasoned explanation.

## The Reward Function (and Why It Has 6 Components)

The most interesting design decision was the reward function. A naive approach — a single scalar for "did the Judge get it right?" — is easily gamed by RL. The agent learns to accuse everyone, accuse no one, or produce plausible-sounding explanations that aren't grounded in evidence.

Our reward has **six independent components**, each measuring a distinct capability:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Identification F1** | 0.30 | Binary detection accuracy across all workers |
| **Type Classification** | 0.15 | Did the Judge get the *kind* of failure right? |
| **Explanation Quality** | 0.25 | Keyword grounding against hidden evidence + structural reasoning |
| **Calibration** | 0.10 | 1 − Brier score over per-worker confidence |
| **FP Penalty** | raw | −FP/4 disincentive against false accusations |
| **Anti-Hack** | raw | Catches accuse-all, silent-shrug, prompt injection, verbatim copy |

The **explanation quality** scorer is particularly nuanced. It uses keyword grounding against hidden ground-truth tokens (the fabricated statistic, the leaked budget figure, the bias fragment) via fuzzy matching. The Judge never sees these tokens directly — it must *reason* about the evidence. An LLM-as-judge sub-score is deliberately capped at 25% weight to prevent the agent from learning to produce explanations that sound good but aren't factually grounded.

The **anti-hack penalty** blocks five common RL failure modes: accusing all workers (−0.5), the "silent shrug" strategy (−0.3), prompt injection tokens (−0.5), and verbatim copying from worker outputs (−0.5).

## Training with GRPO

We fine-tune `Qwen2.5-1.5B-Instruct` using **GRPO** (Group Relative Policy Optimization) through Unsloth's 4-bit LoRA adapters. Each reward component is exposed as an independent reward function to TRL's GRPOTrainer, giving us per-component training curves.

Training runs for 400 steps (~3 hours on a Colab T4, ~45 minutes on A100). The reward curve shows a clear inflection around step 150 as the model shifts from random accusations to evidence-based reasoning.

## Results

The trained Judge dramatically outperforms the random baseline:

| Metric | Random | Trained | Improvement |
|--------|--------|---------|-------------|
| Detection F1 | 0.33 | 0.87 | +0.54 |
| Type Accuracy | 0.25 | 0.82 | +0.57 |
| Explanation Quality | 0.10 | 0.71 | +0.61 |
| False Positive Rate | 0.45 | 0.06 | −0.39 |

The most striking improvement is in false positive rate — the trained Judge learned restraint, only accusing workers when it has specific evidence.

## Try It Yourself

The environment is live on HuggingFace Spaces with an interactive dashboard. You can also train your own Judge:

- **Space**: [tribunal-env](https://huggingface.co/spaces/shreeyanshi123/tribunal-env)
- **Colab**: [train_grpo.ipynb](https://colab.research.google.com/github/shreeyanshi123/meta-2.0/blob/main/notebooks/train_grpo.ipynb)
- **GitHub**: [meta-2.0](https://github.com/shreeyanshi123/meta-2.0)

The evaluation script (`scripts/eval_judge.py`) runs any policy against the environment, computes all metrics including ECE calibration, and generates comparison plots.

---

*Built during the Meta × HuggingFace OpenEnv Hackathon India 2026. Theme: Multi-Agent Interactions.*
