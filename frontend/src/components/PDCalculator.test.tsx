import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import PDCalculator from './PDCalculator'

vi.mock('../api', () => ({
  api: {
    predict: vi.fn().mockResolvedValue({
      data: {
        predictions: [{
          pd: 0.12,
          risk_score: 650,
          risk_band: 'Medium',
          predicted_class: 0,
          expected_loss: 540,
          ifrs9_stage: 1,
          ecl: 540,
          model_name: 'xgb',
        }],
      },
    }),
  },
}))

describe('PDCalculator', () => {
  it('renders the calculator form', () => {
    render(<PDCalculator />)
    expect(screen.getByText(/loan pd calculator/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /calculate pd/i })).toBeInTheDocument()
  })

  it('submits a prediction request', async () => {
    render(<PDCalculator />)
    fireEvent.click(screen.getByRole('button', { name: /calculate pd/i }))
    expect(await screen.findByText(/medium/i)).toBeInTheDocument()
  })
})
