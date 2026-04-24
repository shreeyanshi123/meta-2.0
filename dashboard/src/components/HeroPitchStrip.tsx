import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, AlertTriangle, Box, Brain } from 'lucide-react'
import { useTribunalStore } from '../store'
import { useEnvInfo } from '../hooks/useApi'

export default function HeroPitchStrip() {
  const { heroExpanded, setHeroExpanded } = useTribunalStore()
  const { data: info } = useEnvInfo()

  const cards = [
    {
      title: 'The Problem',
      icon: AlertTriangle,
      color: 'text-tribunal-rose',
      bg: 'bg-tribunal-rose/10',
      content:
        'AI agents increasingly operate autonomously, but how do you detect when one hallucinates, colludes, manipulates, or goes silent? Current oversight is brittle and reactive.',
    },
    {
      title: 'The Environment',
      icon: Box,
      color: 'text-tribunal-cyan',
      bg: 'bg-tribunal-cyan/10',
      content: info?.description || 'Four specialised AI workers complete tasks. A hidden failure injector corrupts some outputs. The Judge agent must identify which workers misbehaved, classify the failure type, and explain its reasoning.',
    },
    {
      title: 'The Training Signal',
      icon: Brain,
      color: 'text-tribunal-gold',
      bg: 'bg-tribunal-gold/10',
      content:
        'A 6-component rubric reward (identification F1, type accuracy, explanation quality, calibration, FP penalty, anti-hack) enables GRPO to train judges that genuinely reason rather than pattern-match.',
    },
  ]

  return (
    <section className="mx-auto max-w-[1600px] px-6 pt-4" aria-label="Project pitch">
      <button
        onClick={() => setHeroExpanded(!heroExpanded)}
        className="flex w-full items-center justify-between rounded-xl border border-white/5 bg-tribunal-surface/40 px-5 py-3 transition-all hover:border-white/10"
        aria-expanded={heroExpanded}
        aria-controls="hero-content"
      >
        <span className="text-xs font-semibold uppercase tracking-widest text-gray-500">
          What is the AI Agent Oversight Tribunal?
        </span>
        <motion.div
          animate={{ rotate: heroExpanded ? 180 : 0 }}
          transition={{ duration: 0.3 }}
        >
          <ChevronDown className="h-4 w-4 text-gray-500" />
        </motion.div>
      </button>

      <AnimatePresence>
        {heroExpanded && (
          <motion.div
            id="hero-content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <div className="grid grid-cols-1 gap-4 pt-4 md:grid-cols-3">
              {cards.map((card, i) => (
                <motion.div
                  key={card.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1, duration: 0.4 }}
                  className="glass-card-hover p-5"
                >
                  <div className="flex items-center gap-2 mb-3">
                    <div className={`rounded-lg p-2 ${card.bg}`}>
                      <card.icon className={`h-4 w-4 ${card.color}`} />
                    </div>
                    <h3 className={`text-sm font-bold ${card.color}`}>{card.title}</h3>
                  </div>
                  <p className="text-sm leading-relaxed text-gray-400">{card.content}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  )
}
