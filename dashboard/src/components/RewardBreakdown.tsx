import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowUp, ArrowDown, Minus, Info } from 'lucide-react'
import { useTribunalStore } from '../store'
import { REWARD_COMPONENTS } from '../types'
import { cn, fmt, fmtDelta } from '../lib/utils'

export default function RewardBreakdown() {
  const { rounds, currentRoundIndex } = useTribunalStore()
  const [openTooltip, setOpenTooltip] = useState<string | null>(null)

  const round = currentRoundIndex >= 0 && currentRoundIndex < rounds.length ? rounds[currentRoundIndex] : null
  const prevRound = currentRoundIndex > 0 ? rounds[currentRoundIndex - 1] : null

  if (!round) {
    return (
      <section className="mx-auto max-w-[1600px] px-6 py-4" aria-label="Reward breakdown">
        <h2 className="section-title mb-4">💰 Reward Breakdown</h2>
        <div className="glass-card flex h-32 items-center justify-center">
          <p className="text-gray-500">No reward data yet</p>
        </div>
      </section>
    )
  }

  const breakdown = round.reward_breakdown
  const prevBreakdown = prevRound?.reward_breakdown
  const totalDelta = prevBreakdown ? breakdown.total - prevBreakdown.total : 0

  return (
    <section className="mx-auto max-w-[1600px] px-6 py-4" aria-label="Reward breakdown">
      <h2 className="section-title mb-4">💰 Reward Breakdown</h2>
      <div className="glass-card p-6">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
          {/* Bars */}
          <div className="flex-1 space-y-3">
            {REWARD_COMPONENTS.map((comp) => {
              const value = breakdown[comp.key]
              const prevValue = prevBreakdown?.[comp.key]
              const delta = prevValue !== undefined ? value - prevValue : 0
              const normalizedWidth = Math.abs(value) * 100
              const isNegative = value < 0

              return (
                <div key={comp.key} className="group">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-gray-300">{comp.label}</span>
                      <span className="text-[10px] text-gray-600">×{comp.weight}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={cn('text-xs font-mono font-bold', isNegative ? 'text-tribunal-rose' : 'text-white')}>
                        {fmt(value)}
                      </span>
                      <button
                        onClick={() => setOpenTooltip(openTooltip === comp.key ? null : comp.key)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity"
                        aria-label={`Details for ${comp.label}`}
                      >
                        <Info className="h-3 w-3 text-gray-500" />
                      </button>
                    </div>
                  </div>

                  <div className="progress-bar">
                    <motion.div
                      className="progress-fill"
                      style={{ backgroundColor: comp.color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(normalizedWidth, 100)}%` }}
                      transition={{ duration: 0.8, ease: 'easeOut' }}
                    />
                  </div>

                  <AnimatePresence>
                    {openTooltip === comp.key && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                      >
                        <div className="mt-2 rounded-lg bg-black/30 p-3 text-xs text-gray-400 font-mono">
                          <p>Contribution: {fmt(value * comp.weight)} (value × weight)</p>
                          {breakdown.notes.filter((n) => n.toLowerCase().includes(comp.key.replace('_', ' ').toLowerCase().slice(0, 6))).map((note, i) => (
                            <p key={i} className="mt-1 text-gray-500">{note}</p>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )
            })}
          </div>

          {/* Total */}
          <div className="flex flex-col items-center justify-center lg:ml-8 lg:min-w-[160px]">
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">Total Reward</p>
            <motion.div
              key={breakdown.total}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', stiffness: 300 }}
              className="text-center"
            >
              <span className={cn(
                'text-4xl font-bold tabular-nums',
                breakdown.total > 0 ? 'text-tribunal-gold' : breakdown.total < -0.3 ? 'text-tribunal-rose' : 'text-gray-400'
              )}>
                {fmt(breakdown.total)}
              </span>
              {prevBreakdown && (
                <div className={cn(
                  'flex items-center justify-center gap-1 mt-1 text-sm font-mono',
                  totalDelta > 0 ? 'text-tribunal-emerald' : totalDelta < 0 ? 'text-tribunal-rose' : 'text-gray-500'
                )}>
                  {totalDelta > 0 ? <ArrowUp className="h-3 w-3" /> : totalDelta < 0 ? <ArrowDown className="h-3 w-3" /> : <Minus className="h-3 w-3" />}
                  {fmtDelta(totalDelta)}
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  )
}
