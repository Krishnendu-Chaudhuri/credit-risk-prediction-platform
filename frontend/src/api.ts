import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || '/api'

const client = axios.create({
  baseURL: API_URL,
  timeout: 120000,
  withCredentials: true,
})

client.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(error),
)

export interface LoanInput {
  person_age: number
  person_income: number
  person_home_ownership: string
  person_emp_length: number
  loan_intent: string
  loan_grade: string
  loan_amnt: number
  loan_int_rate: number
  loan_percent_income: number
  cb_person_default_on_file: string
  cb_person_cred_hist_length: number
}

export interface Prediction {
  pd: number
  risk_score: number
  risk_band: string
  predicted_class: number
  expected_loss: number
  ifrs9_stage: number
  ecl: number
  model_name: string
  calibrated_pd?: number
  optimal_threshold_class?: number
  prediction_id?: string
  shap_contributions?: { feature: string; contribution: number }[]
  reason_codes?: string[]
}

export interface ModelMetrics {
  model_name: string
  roc_auc: number
  f1: number
  precision: number
  recall: number
  accuracy: number
  ks: number
  gini: number
}

export interface StressResult {
  scenario: string
  avg_pd: number
  total_el: number
  total_ecl: number
  stage_1_count: number
  stage_2_count: number
  stage_3_count: number
  pd_multiplier: number
}

export interface TrainResult {
  best_model: string
  metrics: Record<string, ModelMetrics>
  feature_importance: Record<string, number>
  training_metadata: Record<string, unknown>
}

interface TrainJobStatusPayload {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  result?: TrainResult
  error?: string
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

export async function establishSession(apiKey?: string): Promise<void> {
  await client.post('/auth/session', apiKey ? { api_key: apiKey } : {})
}

export async function logoutSession(): Promise<void> {
  await client.post('/auth/logout')
}

async function pollTrainingJob(jobId: string): Promise<TrainResult> {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const statusResp = await client.get<TrainJobStatusPayload>(`/train/status/${jobId}`)
    const { status, result, error } = statusResp.data
    if (status === 'completed' && result) {
      return result
    }
    if (status === 'failed') {
      throw new Error(error || 'Training failed')
    }
    await sleep(1000)
  }
  throw new Error('Training timed out')
}

export const api = {
  health: () => client.get('/health'),
  train: async () => {
    const start = await client.post<{ job_id: string; status: string }>('/train', {})
    return pollTrainingJob(start.data.job_id)
  },
  metrics: () => client.get('/metrics'),
  predict: (loan: LoanInput, model_name = 'xgb') =>
    client.post<{ predictions: Prediction[] }>('/predict', { loan, model_name }),
  stressTest: (sample_size: number, scenarios: string[], model_name = 'xgb') =>
    client.post<{ results: StressResult[]; comparison: unknown[]; loan_count: number }>(
      '/stress_test',
      { sample_size, scenarios, model_name },
    ),
}

export { client }
