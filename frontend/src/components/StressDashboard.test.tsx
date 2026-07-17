import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import StressDashboard from './StressDashboard'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Bar: () => null,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}))

vi.mock('../api', () => ({
  api: {
    stressTest: vi.fn(),
  },
}))

describe('StressDashboard', () => {
  it('renders scenario controls', () => {
    render(<StressDashboard />)
    expect(screen.getByText(/stress test dashboard/i)).toBeInTheDocument()
    expect(screen.getByText('Normal')).toBeInTheDocument()
    expect(screen.getByText('Boom')).toBeInTheDocument()
    expect(screen.getByText('Recession')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /run stress test/i })).toBeInTheDocument()
  })
})
