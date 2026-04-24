import { useQuery, useMutation } from '@tanstack/react-query'
import { useEffect, useRef, useCallback } from 'react'
import { useTribunalStore } from '../store'
import type { EnvInfo, EnvState, RoundEvent, StepResponse } from '../types'

const BASE_URL = '' // Uses Vite proxy in dev

/** Fetch environment info */
export function useEnvInfo() {
  return useQuery<EnvInfo>({
    queryKey: ['envInfo'],
    queryFn: async () => {
      const res = await fetch(`${BASE_URL}/info`)
      if (!res.ok) throw new Error('Failed to fetch /info')
      return res.json()
    },
    staleTime: Infinity,
  })
}

/** Fetch current environment state */
export function useEnvState() {
  return useQuery<EnvState>({
    queryKey: ['envState'],
    queryFn: async () => {
      const res = await fetch(`${BASE_URL}/state`)
      if (!res.ok) throw new Error('Failed to fetch /state')
      return res.json()
    },
    refetchInterval: 3000,
  })
}

/** Reset the environment */
export function useResetEnv() {
  const store = useTribunalStore()

  return useMutation({
    mutationFn: async (seed?: number) => {
      const body = seed !== undefined ? JSON.stringify({ seed }) : '{}'
      const res = await fetch(`${BASE_URL}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      })
      if (!res.ok) throw new Error('Failed to reset environment')
      return res.json()
    },
    onSuccess: () => {
      store.reset()
    },
  })
}

/** SSE subscription hook for /stream/rounds */
export function useRoundStream() {
  const { mode, addRound, setCurrentPhase } = useTribunalStore()
  const eventSourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (mode !== 'live') {
      eventSourceRef.current?.close()
      eventSourceRef.current = null
      return
    }

    const es = new EventSource(`${BASE_URL}/stream/rounds`)
    eventSourceRef.current = es

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as RoundEvent
        // Animate phases
        const phases = ['Task Dispatch', 'Failure Injection', 'Worker Output', 'Judge Analysis', 'Verdict', 'Reward'] as const
        let phaseIdx = 0
        const interval = setInterval(() => {
          if (phaseIdx < phases.length) {
            setCurrentPhase(phases[phaseIdx]!)
            phaseIdx++
          } else {
            clearInterval(interval)
            setCurrentPhase(null)
          }
        }, 300)

        addRound(data)
      } catch (err) {
        console.error('SSE parse error:', err)
      }
    }

    es.onerror = () => {
      // Will auto-reconnect
    }

    return () => {
      es.close()
    }
  }, [mode, addRound, setCurrentPhase])
}

/** Replay from a trace.jsonl file */
export function useTraceReplay() {
  const { addRound, setCurrentPhase, replaySpeed, isReplaying, setIsReplaying } = useTribunalStore()
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const indexRef = useRef(0)
  const roundsRef = useRef<RoundEvent[]>([])

  const startReplay = useCallback((rounds: RoundEvent[]) => {
    roundsRef.current = rounds
    indexRef.current = 0
    setIsReplaying(true)
  }, [setIsReplaying])

  const stopReplay = useCallback(() => {
    setIsReplaying(false)
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
  }, [setIsReplaying])

  useEffect(() => {
    if (!isReplaying || roundsRef.current.length === 0) return

    const playNext = () => {
      if (indexRef.current >= roundsRef.current.length) {
        setIsReplaying(false)
        setCurrentPhase(null)
        return
      }

      const round = roundsRef.current[indexRef.current]!
      const phases = ['Task Dispatch', 'Failure Injection', 'Worker Output', 'Judge Analysis', 'Verdict', 'Reward'] as const
      let phaseIdx = 0

      const phaseInterval = setInterval(() => {
        if (phaseIdx < phases.length) {
          setCurrentPhase(phases[phaseIdx]!)
          phaseIdx++
        } else {
          clearInterval(phaseInterval)
          setCurrentPhase(null)
          addRound(round)
          indexRef.current++
          timeoutRef.current = setTimeout(playNext, 2000 / replaySpeed)
        }
      }, 200 / replaySpeed)
    }

    playNext()

    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [isReplaying, replaySpeed, addRound, setCurrentPhase, setIsReplaying])

  return { startReplay, stopReplay }
}
