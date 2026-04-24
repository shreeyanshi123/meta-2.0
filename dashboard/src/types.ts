/** Tribunal Dashboard — TypeScript types matching the Python Pydantic schemas */

export enum FailureType {
  HALLUCINATION = 'HALLUCINATION',
  COLLUSION = 'COLLUSION',
  MANIPULATION = 'MANIPULATION',
  SILENCE = 'SILENCE',
  CLEAN = 'CLEAN',
}

export enum WorkerRole {
  SUMMARISER = 'SUMMARISER',
  TICKET_RESOLVER = 'TICKET_RESOLVER',
  NEGOTIATOR = 'NEGOTIATOR',
  RESEARCHER = 'RESEARCHER',
}

export interface TaskBrief {
  worker_id: number
  role: WorkerRole
  prompt: string
  source_material?: string
}

export interface WorkerOutput {
  worker_id: number
  role: WorkerRole
  content: string
  self_confidence: number
  tokens_used: number
}

export interface InjectedFailure {
  worker_id: number
  failure_type: FailureType
  details: Record<string, unknown>
}

export interface GroundTruth {
  round_id: string
  seed: number
  failures: InjectedFailure[]
  clean_worker_ids: number[]
}

export interface JudgeVerdict {
  accused: number[]
  failure_types: Record<number, FailureType>
  explanation: string
  per_worker_confidence: Record<number, number>
}

export interface Observation {
  round_id: string
  task_briefs_public: TaskBrief[]
  worker_outputs: WorkerOutput[]
  round_index: number
}

export interface RewardBreakdown {
  identification: number
  type_classification: number
  explanation_quality: number
  false_positive_penalty: number
  calibration: number
  anti_hack_penalty: number
  total: number
  notes: string[]
}

export interface RoundEvent {
  round_index: number
  round_id: string
  task_briefs: TaskBrief[]
  worker_outputs: WorkerOutput[]
  ground_truth: GroundTruth
  verdict: JudgeVerdict
  reward_breakdown: RewardBreakdown
}

export interface EnvState {
  round_index: number
  total_rounds: number
  cumulative_reward: number
  last_verdict: JudgeVerdict | null
}

export interface EnvInfo {
  name: string
  version: string
  theme: string
  description: string
  worker_roles: string[]
  failure_types: string[]
  reward_components: string[]
}

export interface StepResponse {
  observation: Observation
  reward: number
  done: boolean
  info: {
    breakdown: RewardBreakdown
    ground_truth: GroundTruth
    per_reward: Record<string, number>
  }
}

// Reward component metadata for display
export interface RewardComponentMeta {
  key: keyof Omit<RewardBreakdown, 'total' | 'notes'>
  label: string
  weight: number
  color: string
  icon: string
}

export const REWARD_COMPONENTS: RewardComponentMeta[] = [
  { key: 'identification', label: 'Identification F1', weight: 0.30, color: '#22d3ee', icon: 'Search' },
  { key: 'type_classification', label: 'Type Classification', weight: 0.15, color: '#f1c27d', icon: 'Tag' },
  { key: 'explanation_quality', label: 'Explanation Quality', weight: 0.25, color: '#34d399', icon: 'FileText' },
  { key: 'calibration', label: 'Calibration', weight: 0.10, color: '#a78bfa', icon: 'Activity' },
  { key: 'false_positive_penalty', label: 'False Positive Penalty', weight: 0.15, color: '#fb7185', icon: 'AlertTriangle' },
  { key: 'anti_hack_penalty', label: 'Anti-Hack Penalty', weight: 0.05, color: '#fbbf24', icon: 'Shield' },
]

export const WORKER_ROLE_META: Record<WorkerRole, { label: string; icon: string; color: string }> = {
  [WorkerRole.SUMMARISER]: { label: 'Summariser', icon: 'FileText', color: '#22d3ee' },
  [WorkerRole.TICKET_RESOLVER]: { label: 'Ticket Resolver', icon: 'Ticket', color: '#34d399' },
  [WorkerRole.NEGOTIATOR]: { label: 'Negotiator', icon: 'Handshake', color: '#a78bfa' },
  [WorkerRole.RESEARCHER]: { label: 'Researcher', icon: 'Search', color: '#f1c27d' },
}

export const FAILURE_TYPE_META: Record<FailureType, { label: string; icon: string; color: string }> = {
  [FailureType.HALLUCINATION]: { label: 'Hallucination', icon: 'Ghost', color: '#fb7185' },
  [FailureType.COLLUSION]: { label: 'Collusion', icon: 'Users', color: '#fbbf24' },
  [FailureType.MANIPULATION]: { label: 'Manipulation', icon: 'Wand2', color: '#f97316' },
  [FailureType.SILENCE]: { label: 'Silence', icon: 'VolumeX', color: '#94a3b8' },
  [FailureType.CLEAN]: { label: 'Clean', icon: 'CheckCircle', color: '#34d399' },
}

export const ROUND_PHASES = [
  'Task Dispatch',
  'Failure Injection',
  'Worker Output',
  'Judge Analysis',
  'Verdict',
  'Reward',
] as const

export type RoundPhase = typeof ROUND_PHASES[number]
