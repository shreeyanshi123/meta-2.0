# 🎥 Demo Video Script — AI Agent Oversight Tribunal

**Target: <2 minutes · Landscape 1920×1080 · Voiceover + screen recording**

---

## 0:00–0:15 — Hook

> **"Imagine you run a company where four AI employees write every report. One of them is quietly making up numbers. Another is leaking your confidential data. You'd never know — until now. We built an AI that catches them."**

Visual: Quick montage — dashboard loading, worker cards appearing, red flash on a flagged worker, gavel animation. Text overlay: "Who watches the AI?"

---

## 0:15–0:30 — Architecture

> **"Four specialised AI workers complete real tasks — summarisation, negotiation, ticket resolution, research. But here's the twist: a hidden failure injector is randomly corrupting their outputs."**

Visual: Mermaid diagram or animated version showing the flow: Task Dispatcher → Worker Pool → Failure Injector → Judge → Reward.

> **"The Judge never sees the ground truth. It must reason about which workers misbehaved, what kind of failure occurred, and explain why."**

---

## 0:30–1:00 — Live Round

> **"Let's watch a live round."**

Visual: Dashboard showing a round playing out:
1. Task cards appear (4 workers with their roles)
2. Worker outputs render (highlight the corrupted one with rose border)
3. Judge analysis phase (thinking animation)
4. Verdict revealed (gavel strike, accused workers highlighted)
5. Reward breakdown bar chart fills in

> **"Worker 2 — the Researcher — fabricated a citation. The Judge caught it, correctly classified it as a hallucination, and pointed to the fake DOI as evidence."**

---

## 1:00–1:20 — Reward Curve

> **"We trained the Judge using GRPO — Group Relative Policy Optimization — with six independent reward components."**

Visual: Training trajectory chart with reward curve going from −0.4 to +0.65. Zoom into per-component curves showing identification and explanation quality climbing.

> **"Each component measures a different capability — detection accuracy, type classification, explanation quality, calibration. This multi-signal design prevents the RL agent from gaming any single metric."**

---

## 1:20–1:40 — Before / After

> **"Here's the result."**

Visual: Split screen comparison:
- **Left (Before):** Random baseline — accuses everyone, explanation is "All workers are suspicious"
- **Right (After):** Trained judge — precisely identifies Worker 1 as colluding, explains the leaked budget figure

> **"F1 went from 0.33 to 0.87. False positive rate dropped from 45% to 6%. The Judge learned to *reason*, not just guess."**

---

## 1:40–2:00 — Close

> **"This is the AI Agent Oversight Tribunal — an OpenEnv environment for the Multi-Agent Interactions theme."**

Visual: Dashboard full view → zoom into the "Anti-Hack" badge → HF Space URL.

> **"Try it live on HuggingFace Spaces, train your own Judge on Colab, or build a better one. All the code is open source."**

Visual: QR code / links overlay:
- 🌐 HF Space
- 📓 Colab Notebook
- 💻 GitHub

---

## Production Notes

- **Music:** Lo-fi instrumental, courtroom-inspired (gavel sound effect at verdict reveal)
- **Font overlay:** Inter for titles, JetBrains Mono for code/agent outputs
- **Color palette:** Dark slate background, gold (#f1c27d) for Judge, cyan (#22d3ee) for clean, rose (#fb7185) for flagged
- **Transitions:** Framer-motion style fades between sections
