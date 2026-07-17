import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts'
import { api, StressResult } from '../api'

export default function StressDashboard() {
  const [sampleSize, setSampleSize] = useState(500)
  const [scenarios, setScenarios] = useState(['Normal', 'Boom', 'Recession'])
  const [model, setModel] = useState('xgb')
  const [results, setResults] = useState<StressResult[]>([])
  const [loanCount, setLoanCount] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const toggleScenario = (s: string) => {
    setScenarios((prev) => (prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]))
  }

  const runStress = async () => {
    setLoading(true)
    setError('')
    try {
      const { data } = await api.stressTest(sampleSize, scenarios, model)
      setResults(data.results)
      setLoanCount(data.loan_count)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Stress test failed')
    } finally {
      setLoading(false)
    }
  }

  const chartData = results.map((r) => ({
    scenario: r.scenario,
    avg_pd: +(r.avg_pd * 100).toFixed(2),
    total_ecl: +r.total_ecl.toFixed(0),
    stage_1: r.stage_1_count,
    stage_2: r.stage_2_count,
    stage_3: r.stage_3_count,
  }))

  return (
    <div>
      <div className="card">
        <h2>Stress Test Dashboard</h2>
        <label>Portfolio Sample Size</label>
        <input type="number" value={sampleSize} min={100} max={5000} onChange={(e) => setSampleSize(+e.target.value)} />
        <label>Scenarios</label>
        <div style={{ marginBottom: '1rem' }}>
          {['Normal', 'Boom', 'Recession'].map((s) => (
            <label key={s} style={{ display: 'inline-block', marginRight: '1rem' }}>
              <input type="checkbox" checked={scenarios.includes(s)} onChange={() => toggleScenario(s)} /> {s}
            </label>
          ))}
        </div>
        <label>Model</label>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="xgb">XGBoost</option>
          <option value="lr">Logistic Regression</option>
        </select>
        <button className="primary" onClick={runStress} disabled={loading || scenarios.length === 0}>
          {loading ? 'Running...' : 'Run Stress Test'}
        </button>
        {error && <p className="error">{error}</p>}
      </div>

      {results.length > 0 && (
        <>
          <div className="card">
            <p>Loans tested: <strong>{loanCount}</strong></p>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="scenario" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Bar yAxisId="left" dataKey="avg_pd" name="Avg PD (%)" fill="#16213e" />
                <Bar yAxisId="right" dataKey="total_ecl" name="Total ECL ($)" fill="#e94560" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3>IFRS 9 Stage Distribution</h3>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="scenario" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="stage_1" name="Stage 1" stroke="#2ecc71" />
                <Line type="monotone" dataKey="stage_2" name="Stage 2" stroke="#f39c12" />
                <Line type="monotone" dataKey="stage_3" name="Stage 3" stroke="#e74c3c" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  )
}
