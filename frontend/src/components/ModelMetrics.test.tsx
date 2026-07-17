import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import ModelMetricsView from './ModelMetrics'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
}))

vi.mock('../api', () => ({
  api: {
    metrics: vi.fn().mockResolvedValue({
      data: {
        metrics: {
          xgb: {
            model_name: 'xgb',
            roc_auc: 0.8,
            f1: 0.7,
            precision: 0.7,
            recall: 0.7,
            accuracy: 0.8,
            ks: 0.4,
            gini: 0.6,
          },
        },
        feature_importance: { xgb: { loan_amnt: 0.5 } },
      },
    }),
    train: vi.fn(),
  },
}))

describe('ModelMetricsView', () => {
  it('renders metrics section after load', async () => {
    render(<ModelMetricsView />)
    await waitFor(() => {
      expect(screen.getByText(/model metrics & feature importance/i)).toBeInTheDocument()
    })
    expect(screen.getByRole('button', { name: /retrain models/i })).toBeInTheDocument()
  })
})
