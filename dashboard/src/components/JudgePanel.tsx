import { motion } from 'framer-motion'
import { Gavel, AlertCircle } from 'lucide-react'
import { useTribunalStore } from '../store'
import { cn } from '../lib/utils'
import { highlightKeywords } from '../lib/utils'
import { FAILURE_TYPE_META } from '../types'
import type { FailureType } from '../types'

export default function JudgePanel() {
  const { rounds, currentRoundIndex } = useTribunalStore()
  const round = currentRoundIndex >= 0 && currentRoundIndex < rounds.length ? rounds[currentRoundIndex] : null

  if (!round || !round.verdict) {
    return (
      <section className="mx-auto max-w-[1600px] px-6 py-4" aria-label="Judge panel">
        <h2 className="section-title mb-4">⚖️ Judge Panel</h2>
        <div className="glass-card flex h-40 items-center justify-center">
          <p className="text-gray-500">No verdict yet</p>
        </div>
      </section>
    )
  }

  const { verdict, ground_truth } = round
  const gtKeywords = ground_truth.failures.flatMap((f) =>
    Object.values(f.details).filter((v): v is string => typeof v === 'string').flatMap((s) => s.split(/\s+/).filter((w) => w.length > 4))
  )

  return (
    <section className="mx-auto max-w-[1600px] px-6 py-4" aria-label="Judge panel">
      <h2 className="section-title mb-4">⚖️ Judge Panel</h2>
      <motion.div
        className="glass-card p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-3 mb-5">
          <motion.div
            className="rounded-xl bg-tribunal-gold/15 p-3"
            animate={{ rotate: [0, -15, 5, 0] }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
          >
            <Gavel className="h-6 w-6 text-tribunal-gold" />
          </motion.div>
          <div>
            <h3 className="text-lg font-bold text-white">Judge Verdict</h3>
            <p className="text-xs text-gray-500">Round {round.round_index + 1}</p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Accused list */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">Accused Workers</p>
            <div className="flex flex-wrap gap-2">
              {verdict.accused.length === 0 ? (
                <span className="badge-cyan">All Clear</span>
              ) : (
                verdict.accused.map((wid) => (
                  <motion.span key={wid} className="badge-rose" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring' }}>
                    <AlertCircle className="h-3 w-3" />
                    Worker {wid}
                  </motion.span>
                ))
              )}
            </div>

            {/* Failure types */}
            {verdict.accused.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">Failure Types</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(verdict.failure_types).map(([wid, ftype]) => {
                    const meta = FAILURE_TYPE_META[ftype as FailureType]
                    return (
                      <span key={wid} className="badge" style={{ backgroundColor: `${meta?.color}20`, color: meta?.color, border: `1px solid ${meta?.color}40` }}>
                        W{wid}: {meta?.label ?? ftype}
                      </span>
                    )
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Confidence bars */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">Per-Worker Confidence</p>
            <div className="space-y-2">
              {Object.entries(verdict.per_worker_confidence).map(([wid, conf]) => {
                const isAccused = verdict.accused.includes(Number(wid))
                return (
                  <div key={wid} className="flex items-center gap-3">
                    <span className="text-xs text-gray-400 w-6 font-mono">W{wid}</span>
                    <div className="progress-bar flex-1">
                      <motion.div
                        className={cn('progress-fill', isAccused ? 'bg-tribunal-rose' : 'bg-tribunal-cyan')}
                        initial={{ width: 0 }}
                        animate={{ width: `${conf * 100}%` }}
                        transition={{ duration: 0.8, ease: 'easeOut' }}
                      />
                    </div>
                    <span className="text-xs font-mono text-gray-400 w-10 text-right">{(conf * 100).toFixed(0)}%</span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Explanation */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">Explanation</p>
            <div
              className="rounded-lg bg-black/20 p-3 text-sm leading-relaxed text-gray-300 font-mono"
              dangerouslySetInnerHTML={{ __html: highlightKeywords(verdict.explanation, gtKeywords.slice(0, 10)) }}
            />
          </div>
        </div>
      </motion.div>
    </section>
  )
}
