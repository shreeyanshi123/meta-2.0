import { create } from 'zustand'
import type { RoundEvent, RoundPhase } from './types'

interface TribunalStore {
  // Connection mode
  mode: 'live' | 'replay'
  setMode: (mode: 'live' | 'replay') => void

  // Seed
  seed: number
  setSeed: (seed: number) => void

  // Round data
  rounds: RoundEvent[]
  currentRoundIndex: number
  addRound: (round: RoundEvent) => void
  setRounds: (rounds: RoundEvent[]) => void
  setCurrentRoundIndex: (index: number) => void

  // Current phase animation
  currentPhase: RoundPhase | null
  setCurrentPhase: (phase: RoundPhase | null) => void

  // Replay controls
  replaySpeed: number
  setReplaySpeed: (speed: number) => void
  isReplaying: boolean
  setIsReplaying: (playing: boolean) => void

  // UI state
  heroExpanded: boolean
  setHeroExpanded: (expanded: boolean) => void

  // Total rounds from env state
  totalRounds: number
  setTotalRounds: (total: number) => void

  // Reset everything
  reset: () => void
}

export const useTribunalStore = create<TribunalStore>((set) => ({
  mode: 'live',
  setMode: (mode) => set({ mode }),

  seed: 42,
  setSeed: (seed) => set({ seed }),

  rounds: [],
  currentRoundIndex: -1,
  addRound: (round) =>
    set((state) => ({
      rounds: [...state.rounds, round],
      currentRoundIndex: state.rounds.length,
    })),
  setRounds: (rounds) => set({ rounds, currentRoundIndex: rounds.length > 0 ? rounds.length - 1 : -1 }),
  setCurrentRoundIndex: (index) => set({ currentRoundIndex: index }),

  currentPhase: null,
  setCurrentPhase: (phase) => set({ currentPhase: phase }),

  replaySpeed: 1,
  setReplaySpeed: (speed) => set({ replaySpeed: speed }),
  isReplaying: false,
  setIsReplaying: (playing) => set({ isReplaying: playing }),

  heroExpanded: false,
  setHeroExpanded: (expanded) => set({ heroExpanded: expanded }),

  totalRounds: 5,
  setTotalRounds: (total) => set({ totalRounds: total }),

  reset: () =>
    set({
      rounds: [],
      currentRoundIndex: -1,
      currentPhase: null,
      isReplaying: false,
    }),
}))
