import { useCallback, useRef } from 'react'
import { Upload, Play, Pause, FastForward } from 'lucide-react'
import { motion } from 'framer-motion'
import { useTribunalStore } from '../store'
import { useTraceReplay } from '../hooks/useApi'
import { parseTraceFile } from '../lib/utils'
import { cn } from '../lib/utils'
import type { RoundEvent } from '../types'

export default function ReplayControls() {
  const { mode, isReplaying, replaySpeed, setReplaySpeed, reset, setMode } = useTribunalStore()
  const { startReplay, stopReplay } = useTraceReplay()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (!file) return

      const reader = new FileReader()
      reader.onload = (ev) => {
        const content = ev.target?.result as string
        const parsed = parseTraceFile(content) as RoundEvent[]
        reset()
        setMode('replay')
        startReplay(parsed)
      }
      reader.readAsText(file)
    },
    [reset, setMode, startReplay]
  )

  if (mode !== 'replay') return null

  return (
    <div className="mx-auto max-w-[1600px] px-6 py-2">
      <div className="glass-card flex items-center gap-4 px-5 py-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-tribunal-amber">
          Replay Mode
        </span>

        {/* File upload */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".jsonl,.json"
          onChange={handleFileUpload}
          className="hidden"
          id="trace-file-input"
          aria-label="Upload trace file"
        />
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center gap-2 rounded-lg bg-white/5 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-white/10 transition-all"
        >
          <Upload className="h-3 w-3" />
          Load trace.jsonl
        </motion.button>

        {/* Play / Pause */}
        <button
          onClick={() => (isReplaying ? stopReplay() : null)}
          className={cn(
            'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all',
            isReplaying ? 'bg-tribunal-amber/20 text-tribunal-amber' : 'bg-white/5 text-gray-400'
          )}
          aria-label={isReplaying ? 'Pause replay' : 'Play replay'}
        >
          {isReplaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
          {isReplaying ? 'Pause' : 'Play'}
        </button>

        {/* Speed control */}
        <div className="flex items-center gap-2">
          <FastForward className="h-3 w-3 text-gray-500" />
          {[0.5, 1, 2, 4].map((speed) => (
            <button
              key={speed}
              onClick={() => setReplaySpeed(speed)}
              className={cn(
                'rounded px-2 py-1 text-[10px] font-mono transition-all',
                replaySpeed === speed ? 'bg-tribunal-amber/20 text-tribunal-amber' : 'text-gray-500 hover:text-gray-300'
              )}
              aria-label={`Set replay speed to ${speed}x`}
            >
              {speed}×
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
