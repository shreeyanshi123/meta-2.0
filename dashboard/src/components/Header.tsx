import { Gavel, Radio, RotateCcw } from 'lucide-react'
import { motion } from 'framer-motion'
import { useTribunalStore } from '../store'
import { useResetEnv } from '../hooks/useApi'
import { cn } from '../lib/utils'

export default function Header() {
  const { mode, setMode, seed, setSeed, rounds, totalRounds } = useTribunalStore()
  const resetMutation = useResetEnv()

  const currentRound = rounds.length
  const displayRound = currentRound > 0 ? currentRound : 0

  return (
    <header
      className="sticky top-0 z-50 border-b border-white/10 bg-tribunal-bg/90 backdrop-blur-xl"
      role="banner"
    >
      <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-3">
        {/* Left — Wordmark */}
        <div className="flex items-center gap-3">
          <motion.div
            whileHover={{ rotate: -15, scale: 1.1 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <Gavel className="h-7 w-7 text-tribunal-gold" aria-hidden="true" />
          </motion.div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white">
              AI Agent Oversight <span className="gold-text">Tribunal</span>
            </h1>
            <p className="text-[10px] font-medium uppercase tracking-widest text-gray-500">
              OpenEnv Hackathon — Meta × HuggingFace
            </p>
          </div>
        </div>

        {/* Center — Round Counter */}
        <motion.div
          className="glass-card px-6 py-2 text-center"
          key={displayRound}
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 400, damping: 20 }}
        >
          <span className="text-xs font-medium uppercase tracking-wider text-gray-400">Round</span>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold tabular-nums text-tribunal-gold">{displayRound}</span>
            <span className="text-sm text-gray-500">/</span>
            <span className="text-sm tabular-nums text-gray-400">{totalRounds}</span>
          </div>
        </motion.div>

        {/* Right — Controls */}
        <div className="flex items-center gap-3">
          {/* Live / Replay Toggle */}
          <div className="glass-card flex items-center gap-1 p-1" role="radiogroup" aria-label="Mode selection">
            <button
              className={cn(
                'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all',
                mode === 'live'
                  ? 'bg-tribunal-cyan/20 text-tribunal-cyan'
                  : 'text-gray-400 hover:text-gray-200'
              )}
              onClick={() => setMode('live')}
              role="radio"
              aria-checked={mode === 'live'}
              aria-label="Live mode"
            >
              <Radio className="h-3 w-3" />
              Live
            </button>
            <button
              className={cn(
                'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all',
                mode === 'replay'
                  ? 'bg-tribunal-amber/20 text-tribunal-amber'
                  : 'text-gray-400 hover:text-gray-200'
              )}
              onClick={() => setMode('replay')}
              role="radio"
              aria-checked={mode === 'replay'}
              aria-label="Replay mode"
            >
              ▶ Replay
            </button>
          </div>

          {/* Seed Input */}
          <div className="glass-card flex items-center gap-2 px-3 py-1.5">
            <label htmlFor="seed-input" className="text-xs text-gray-400">
              Seed
            </label>
            <input
              id="seed-input"
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
              className="w-16 bg-transparent text-sm font-mono text-white outline-none"
              aria-label="Random seed"
            />
          </div>

          {/* New Round Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => resetMutation.mutate(seed)}
            disabled={resetMutation.isPending}
            className={cn(
              'flex items-center gap-2 rounded-lg bg-gradient-to-r from-tribunal-gold to-tribunal-gold-dim px-4 py-2',
              'text-sm font-semibold text-tribunal-bg transition-all',
              'hover:shadow-glow-gold disabled:opacity-50'
            )}
            aria-label="Start new round"
          >
            <RotateCcw className={cn('h-4 w-4', resetMutation.isPending && 'animate-spin')} />
            New Round
          </motion.button>
        </div>
      </div>
    </header>
  )
}
