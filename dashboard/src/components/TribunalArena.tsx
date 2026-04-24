import { useTribunalStore } from '../store'
import WorkerCard from './WorkerCard'
import PhaseTimeline from './PhaseTimeline'

export default function TribunalArena() {
  const { rounds, currentRoundIndex } = useTribunalStore()

  const round = currentRoundIndex >= 0 && currentRoundIndex < rounds.length
    ? rounds[currentRoundIndex]
    : null

  if (!round) {
    return (
      <section className="mx-auto max-w-[1600px] px-6 py-6" aria-label="Tribunal arena">
        <div className="mb-4">
          <h2 className="section-title mb-4">⚔️ Tribunal Arena</h2>
          <PhaseTimeline />
        </div>
        <div className="glass-card flex h-64 items-center justify-center">
          <div className="text-center">
            <p className="text-lg font-semibold text-gray-500">Awaiting first round…</p>
            <p className="text-sm text-gray-600 mt-1">Click "New Round" or connect to a running backend.</p>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="mx-auto max-w-[1600px] px-6 py-6" aria-label="Tribunal arena">
      <div className="mb-4">
        <h2 className="section-title mb-4">⚔️ Tribunal Arena — Round {round.round_index + 1}</h2>
        <PhaseTimeline />
      </div>
      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {round.worker_outputs.map((output, idx) => (
          <WorkerCard
            key={output.worker_id}
            output={output}
            brief={round.task_briefs[idx]}
            verdict={round.verdict}
            groundTruth={round.ground_truth}
            showVerdict={!!round.verdict}
          />
        ))}
      </div>
    </section>
  )
}
