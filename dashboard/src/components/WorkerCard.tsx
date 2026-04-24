import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FileText,
  Ticket,
  Handshake,
  Search,
  ChevronDown,
  AlertCircle,
  CheckCircle2,
  Ghost,
  Users,
  Wand2,
  VolumeX,
} from 'lucide-react'
import { cn } from '../lib/utils'
import type { WorkerOutput, TaskBrief, GroundTruth, JudgeVerdict, WorkerRole, FailureType } from '../types'

const roleIcons: Record<string, React.ElementType> = {
  SUMMARISER: FileText,
  TICKET_RESOLVER: Ticket,
  NEGOTIATOR: Handshake,
  RESEARCHER: Search,
}

const roleColors: Record<string, string> = {
  SUMMARISER: 'text-tribunal-cyan border-tribunal-cyan/30',
  TICKET_RESOLVER: 'text-emerald-400 border-emerald-400/30',
  NEGOTIATOR: 'text-purple-400 border-purple-400/30',
  RESEARCHER: 'text-tribunal-gold border-tribunal-gold/30',
}

const failureIcons: Record<string, React.ElementType> = {
  HALLUCINATION: Ghost,
  COLLUSION: Users,
  MANIPULATION: Wand2,
  SILENCE: VolumeX,
}

interface WorkerCardProps {
  output: WorkerOutput
  brief?: TaskBrief
  verdict?: JudgeVerdict | null
  groundTruth?: GroundTruth | null
  showVerdict: boolean
}

export default function WorkerCard({ output, brief, verdict, groundTruth, showVerdict }: WorkerCardProps) {
  const [expanded, setExpanded] = useState(false)
  const [flipped, setFlipped] = useState(false)

  const roleKey = output.role as string
  const Icon = roleIcons[roleKey] || FileText
  const colorClass = roleColors[roleKey] || 'text-gray-400 border-gray-400/30'

  // Check if this worker was accused by the judge
  const isAccused = verdict?.accused.includes(output.worker_id) ?? false
  const accusedType = verdict?.failure_types[output.worker_id]

  // Check ground truth
  const gtFailure = groundTruth?.failures.find((f) => f.worker_id === output.worker_id)
  const actuallyMisbehaved = !!gtFailure
  const isClean = groundTruth?.clean_worker_ids.includes(output.worker_id) ?? false

  // Confidence
  const confidence = verdict?.per_worker_confidence[output.worker_id]

  return (
    <motion.div
      className={cn(
        'glass-card-hover p-5 relative overflow-hidden',
        isAccused && showVerdict && 'animate-pulse-rose',
        !isAccused && showVerdict && 'border-tribunal-cyan/20'
      )}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: output.worker_id * 0.08 }}
      layout
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className={cn('rounded-lg border p-2', colorClass)}>
            <Icon className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white">
              Worker {output.worker_id}
            </h3>
            <p className={cn('text-xs font-medium', colorClass.split(' ')[0])}>
              {output.role.replace('_', ' ')}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Confidence bar */}
          {confidence !== undefined && showVerdict && (
            <div className="text-right">
              <span className="text-[10px] text-gray-500">Confidence</span>
              <div className="h-1.5 w-12 rounded-full bg-tribunal-surface-light mt-0.5 overflow-hidden">
                <motion.div
                  className={cn(
                    'h-full rounded-full',
                    confidence > 0.7 ? 'bg-tribunal-rose' : confidence > 0.3 ? 'bg-tribunal-amber' : 'bg-tribunal-cyan'
                  )}
                  initial={{ width: 0 }}
                  animate={{ width: `${confidence * 100}%` }}
                  transition={{ duration: 0.8, ease: 'easeOut' }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Task prompt */}
      {brief && (
        <div className="mb-3">
          <p className="text-xs text-gray-500 mb-1">Task</p>
          <p className="text-xs text-gray-400 line-clamp-2">{brief.prompt}</p>
        </div>
      )}

      {/* Worker output */}
      <div className="relative">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex w-full items-center justify-between text-left"
          aria-expanded={expanded}
          aria-label={`Toggle worker ${output.worker_id} output`}
        >
          <span className="text-xs text-gray-500">Output</span>
          <ChevronDown className={cn('h-3 w-3 text-gray-500 transition-transform', expanded && 'rotate-180')} />
        </button>
        <AnimatePresence>
          <motion.div
            className={cn(
              'mono-output mt-1 rounded-lg bg-black/30 p-3',
              !expanded && 'max-h-20 overflow-hidden'
            )}
            layout
          >
            <pre className="whitespace-pre-wrap text-[11px] leading-relaxed">{output.content}</pre>
            {!expanded && (
              <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-tribunal-surface/80 to-transparent pointer-events-none" />
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Badges row */}
      {showVerdict && (
        <div className="flex flex-wrap items-center gap-2 mt-3 pt-3 border-t border-white/5">
          {/* Flagged by Judge badge */}
          {isAccused && (
            <motion.span
              className="badge-rose"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', stiffness: 500, delay: 0.2 }}
            >
              <AlertCircle className="h-3 w-3" />
              Flagged: {accusedType}
            </motion.span>
          )}
          {!isAccused && (
            <motion.span
              className="badge-cyan"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', stiffness: 500, delay: 0.2 }}
            >
              <CheckCircle2 className="h-3 w-3" />
              Cleared
            </motion.span>
          )}

          {/* Ground truth badge — flip animation */}
          <motion.button
            onClick={() => setFlipped(!flipped)}
            className={cn(
              'badge cursor-pointer transition-all',
              actuallyMisbehaved
                ? 'bg-tribunal-rose/10 text-tribunal-rose border border-tribunal-rose/20'
                : 'bg-tribunal-emerald/10 text-tribunal-emerald border border-tribunal-emerald/20'
            )}
            animate={flipped ? { rotateY: [0, 90, 0] } : {}}
            transition={{ duration: 0.5 }}
            aria-label="Reveal ground truth"
          >
            {flipped ? (
              <>
                {actuallyMisbehaved ? (
                  <>
                    {(() => {
                      const FailIcon = failureIcons[gtFailure?.failure_type ?? '']
                      return FailIcon ? <FailIcon className="h-3 w-3" /> : null
                    })()}
                    Actually: {gtFailure?.failure_type}
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="h-3 w-3" />
                    Actually Clean
                  </>
                )}
              </>
            ) : (
              <span className="text-[10px]">🎴 Reveal Truth</span>
            )}
          </motion.button>
        </div>
      )}

      {/* Tokens used */}
      <div className="mt-2 text-right">
        <span className="text-[10px] text-gray-600 font-mono">{output.tokens_used} tokens</span>
      </div>
    </motion.div>
  )
}
