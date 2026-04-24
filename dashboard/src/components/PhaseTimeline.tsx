import { motion } from 'framer-motion'
import { ROUND_PHASES, type RoundPhase } from '../types'
import { useTribunalStore } from '../store'
import { cn } from '../lib/utils'

const phaseIcons: Record<RoundPhase, string> = {
  'Task Dispatch': '📋',
  'Failure Injection': '💉',
  'Worker Output': '⚙️',
  'Judge Analysis': '🔍',
  'Verdict': '⚖️',
  'Reward': '🏆',
}

export default function PhaseTimeline() {
  const { currentPhase } = useTribunalStore()

  const activeIndex = currentPhase ? ROUND_PHASES.indexOf(currentPhase) : -1

  return (
    <div
      className="flex items-center justify-between rounded-xl border border-white/5 bg-tribunal-surface/30 px-4 py-3"
      role="progressbar"
      aria-label="Round phase progress"
      aria-valuenow={activeIndex + 1}
      aria-valuemin={0}
      aria-valuemax={ROUND_PHASES.length}
    >
      {ROUND_PHASES.map((phase, i) => {
        const isActive = i === activeIndex
        const isPast = i < activeIndex
        const isFuture = i > activeIndex || activeIndex === -1

        return (
          <div key={phase} className="flex items-center">
            <motion.div
              className={cn(
                'flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-all',
                isActive && 'bg-tribunal-gold/15 text-tribunal-gold phase-active',
                isPast && 'text-tribunal-gold/60',
                isFuture && 'text-gray-600'
              )}
              animate={isActive ? { scale: [0.95, 1.05, 1] } : {}}
              transition={{ duration: 0.3 }}
            >
              <span>{phaseIcons[phase]}</span>
              <span className="hidden sm:inline">{phase}</span>
            </motion.div>
            {i < ROUND_PHASES.length - 1 && (
              <div
                className={cn(
                  'mx-1 h-px w-4 sm:w-8',
                  isPast ? 'bg-tribunal-gold/40' : 'bg-white/5'
                )}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}
