import { useMemo, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { useTribunalStore } from '../store'
import { movingAverage, cn } from '../lib/utils'
import { REWARD_COMPONENTS } from '../types'

const SERIES_CONFIG = [
  { key: 'total', label: 'Total', color: '#f1c27d', strokeWidth: 3 },
  ...REWARD_COMPONENTS.map((c) => ({
    key: c.key,
    label: c.label,
    color: c.color,
    strokeWidth: 1.5,
  })),
]

export default function TrainingTrajectory() {
  const { rounds } = useTribunalStore()
  const [hiddenSeries, setHiddenSeries] = useState<Set<string>>(new Set())

  const chartData = useMemo(() => {
    return rounds.map((r, i) => ({
      step: i + 1,
      total: r.reward_breakdown.total,
      identification: r.reward_breakdown.identification,
      type_classification: r.reward_breakdown.type_classification,
      explanation_quality: r.reward_breakdown.explanation_quality,
      calibration: r.reward_breakdown.calibration,
      false_positive_penalty: r.reward_breakdown.false_positive_penalty,
      anti_hack_penalty: r.reward_breakdown.anti_hack_penalty,
    }))
  }, [rounds])

  const maData = useMemo(() => {
    const totals = rounds.map((r) => r.reward_breakdown.total)
    const ma = movingAverage(totals, Math.min(20, totals.length))
    return ma.map((v, i) => ({ step: i + 1, ma_total: v }))
  }, [rounds])

  const toggleSeries = (key: string) => {
    setHiddenSeries((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  if (rounds.length === 0) {
    return (
      <section className="mx-auto max-w-[1600px] px-6 py-4" aria-label="Training trajectory">
        <h2 className="section-title mb-4">📈 Training Trajectory</h2>
        <div className="glass-card flex h-64 items-center justify-center">
          <p className="text-gray-500">No data to plot yet</p>
        </div>
      </section>
    )
  }

  return (
    <section className="mx-auto max-w-[1600px] px-6 py-4" aria-label="Training trajectory">
      <h2 className="section-title mb-4">📈 Training Trajectory</h2>

      {/* Main chart */}
      <div className="glass-card p-6">
        <div className="mb-4 flex flex-wrap gap-2">
          {SERIES_CONFIG.map((s) => (
            <button
              key={s.key}
              onClick={() => toggleSeries(s.key)}
              className={cn(
                'badge text-[10px] cursor-pointer transition-all',
                hiddenSeries.has(s.key) ? 'opacity-30' : 'opacity-100'
              )}
              style={{
                backgroundColor: `${s.color}15`,
                color: s.color,
                borderColor: `${s.color}40`,
                borderWidth: 1,
              }}
              aria-label={`Toggle ${s.label} series`}
            >
              <div className="h-2 w-2 rounded-full" style={{ backgroundColor: s.color }} />
              {s.label}
            </button>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="step"
              stroke="#475569"
              tick={{ fill: '#64748b', fontSize: 11 }}
              label={{ value: 'Round', position: 'insideBottomRight', offset: -5, fill: '#64748b', fontSize: 11 }}
            />
            <YAxis
              domain={[-1, 1]}
              stroke="#475569"
              tick={{ fill: '#64748b', fontSize: 11 }}
              label={{ value: 'Reward', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0f172a',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                fontSize: '11px',
                fontFamily: 'JetBrains Mono, monospace',
              }}
              labelStyle={{ color: '#94a3b8' }}
            />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />
            {SERIES_CONFIG.map((s) =>
              !hiddenSeries.has(s.key) ? (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  stroke={s.color}
                  strokeWidth={s.strokeWidth}
                  dot={false}
                  activeDot={{ r: 4, fill: s.color }}
                />
              ) : null
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Moving average chart */}
      {rounds.length > 3 && (
        <div className="glass-card mt-4 p-6">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">
            Moving Average (window=20) — Total Reward
          </h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={maData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="step" stroke="#475569" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis domain={[-1, 1]} stroke="#475569" tick={{ fill: '#64748b', fontSize: 11 }} />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="ma_total" stroke="#f1c27d" strokeWidth={2.5} dot={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  fontSize: '11px',
                  fontFamily: 'JetBrains Mono, monospace',
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  )
}
