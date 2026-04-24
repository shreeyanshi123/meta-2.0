import { useEffect } from 'react'
import { motion } from 'framer-motion'
import Header from './components/Header'
import HeroPitchStrip from './components/HeroPitchStrip'
import TribunalArena from './components/TribunalArena'
import JudgePanel from './components/JudgePanel'
import RewardBreakdownSection from './components/RewardBreakdown'
import TrainingTrajectory from './components/TrainingTrajectory'
import BeforeAfterShowcase from './components/BeforeAfterShowcase'
import ReplayControls from './components/ReplayControls'
import { useRoundStream, useEnvState } from './hooks/useApi'
import { useTribunalStore } from './store'

export default function App() {
  // Subscribe to SSE stream in live mode
  useRoundStream()

  // Sync env state
  const { data: envState } = useEnvState()
  const { setTotalRounds } = useTribunalStore()

  useEffect(() => {
    if (envState?.total_rounds) {
      setTotalRounds(envState.total_rounds)
    }
  }, [envState, setTotalRounds])

  return (
    <div className="min-h-screen bg-tribunal-gradient">
      <Header />
      <ReplayControls />

      <main className="pb-16">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
        >
          {/* Section 1: Hero Pitch Strip */}
          <HeroPitchStrip />

          {/* Section 2: Tribunal Arena */}
          <TribunalArena />

          {/* Section 3: Judge Panel */}
          <JudgePanel />

          {/* Section 4: Reward Breakdown */}
          <RewardBreakdownSection />

          {/* Section 5: Training Trajectory */}
          <TrainingTrajectory />

          {/* Section 6: Before / After Showcase */}
          <BeforeAfterShowcase />
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-6 text-center">
        <p className="text-xs text-gray-600">
          AI Agent Oversight Tribunal · OpenEnv Hackathon India 2026 · Meta × HuggingFace
        </p>
      </footer>
    </div>
  )
}
