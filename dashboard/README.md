# AI Agent Oversight Tribunal — Dashboard

> Premium, demo-ready React dashboard for monitoring the AI Agent Oversight Tribunal environment in real time.

![Status: WIP](https://img.shields.io/badge/Status-WIP-yellow)

## Stack

| Layer         | Technology                                     |
|---------------|-------------------------------------------------|
| Framework     | Vite + React 18 + TypeScript                    |
| Styling       | TailwindCSS + shadcn/ui (Radix primitives)      |
| Icons         | lucide-react                                     |
| Animations    | framer-motion                                    |
| Charts        | recharts                                         |
| State         | zustand                                          |
| Server Sync   | TanStack Query + SSE (`/stream/rounds`)          |
| Typography    | Inter + JetBrains Mono (Google Fonts)            |

## Getting Started

### Prerequisites

- Node.js ≥ 18
- npm ≥ 9
- A running backend at `http://localhost:8000` (the FastAPI server)

### Install & Run

```bash
cd dashboard
npm install
npm run dev
```

The dev server starts at `http://localhost:5173` with Vite's proxy forwarding all API calls to `http://localhost:8000`.

### Production Build

```bash
npm run build
```

Output goes to `dashboard/dist/`. The FastAPI server automatically serves this directory at `/dashboard`.

## Architecture

The dashboard connects to the backend in two ways:

1. **REST API** — `POST /reset`, `GET /state`, `GET /info` for one-off queries
2. **Server-Sent Events** — `GET /stream/rounds` for real-time round data pushed from the server

### Replay Mode

For demo videos without a live GPU, use **Replay Mode**:

1. Switch to "Replay" in the header
2. Upload a `trace.jsonl` file (one JSON object per line, each matching the `RoundEvent` schema)
3. Control playback speed with the 0.5×—4× buttons

## Layout

The dashboard is a single-page app with 6 vertically-stacked sections:

1. **Header** — Wordmark, round counter, mode toggle, seed input, New Round button
2. **Hero Pitch Strip** — Collapsible 3-card explanation of the project
3. **Tribunal Arena** — Phase timeline + 4 worker cards (2×2 → 4×1 responsive)
4. **Judge Panel** — Gavel icon, accused chips, failure types, confidence bars, explanation with keyword highlighting
5. **Reward Breakdown** — 6 progress bars with tooltips + total with delta arrow
6. **Training Trajectory** — Multi-line recharts with togglable legend + moving average sub-chart

## Design Language

- **Theme**: Dark courtroom-inspired (`#0b1020 → #0f172a` gradient)
- **Accent Colors**: Gold `#f1c27d` (Judge), Cyan `#22d3ee` (clean), Rose `#fb7185` (flagged), Amber `#fbbf24` (suspicion)
- **Glass Cards**: `border-white/10`, `backdrop-blur-xl`, soft shadow
- **Micro-interactions**: Scale 1.01 on hover, gavel-strike animation, rose pulse on flagged cards, flip animation for ground truth reveal

## Screenshot Capture

```bash
python scripts/capture_dashboard.py
# Saves to assets/dashboard.png at 1440×900
```

## Accessibility

- All interactive elements are keyboard-reachable
- ARIA labels on live regions, toggle buttons, and expandable content
- Semantic HTML with proper heading hierarchy
- Single `<h1>` in the header
