import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { api, ModelMetrics as Metrics } from '../api'

export default function ModelMetricsView() {
  const [metrics, setMetrics] = useState<Record<string, Metrics>>({})
  const [importance, setImportance] = useState<Record<string, Record<string, number>>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selectedModel, setSelectedModel] = useState('xgb')

  useEffect(() => {
    const load = async () => {
      try {
        const { data } = await api.metrics()
        setMetrics(data.metrics)
        setImportance(data.feature_importance || {})
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Failed to load metrics. Train models first.')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const handleTrain = async () => {
    setLoading(true)
    setError('')
    try {
      await api.train()
      const { data } = await api.metrics()
      setMetrics(data.metrics)
      setImportance(data.feature_importance || {})
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Training failed')
    } finally {
      setLoading(false)
    }
  }

  const impData = Object.entries(importance[selectedModel] || importance.xgb || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([name, value]) => ({ name: name.replace(/__/g, ' ').slice(0, 30), value: +value.toFixed(4) }))

  if (loading) return <div className="card">Loading metrics...</div>

  return (
    <div>
      <div className="card">
        <h2>Model Metrics & Feature Importance</h2>
        <button className="primary" onClick={handleTrain}>Retrain Models</button>
        {error && <p className="error">{error}</p>}
      </div>

      {Object.entries(metrics).map(([name, m]) => (
        <div className="card" key={name}>
          <h3>{name.toUpperCase()}</h3>
          <div className="metrics-row">
            <div className="metric"><div className="value">{m.roc_auc.toFixed(4)}</div><div className="label">ROC AUC</div></div>
            <div className="metric"><div className="value">{m.f1.toFixed(4)}</div><div className="label">F1</div></div>
            <div className="metric"><div className="value">{m.precision.toFixed(4)}</div><div className="label">Precision</div></div>
            <div className="metric"><div className="value">{m.recall.toFixed(4)}</div><div className="label">Recall</div></div>
            <div className="metric"><div className="value">{m.accuracy.toFixed(4)}</div><div className="label">Accuracy</div></div>
            <div className="metric"><div className="value">{m.ks.toFixed(4)}</div><div className="label">KS</div></div>
            <div className="metric"><div className="value">{m.gini.toFixed(4)}</div><div className="label">Gini</div></div>
          </div>
        </div>
      ))}

      {impData.length > 0 && (
        <div className="card">
          <h3>Feature Importance</h3>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            <option value="xgb">XGBoost</option>
            <option value="lr">Logistic Regression</option>
          </select>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={impData} layout="vertical" margin={{ left: 120 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={110} />
              <Tooltip />
              <Bar dataKey="value" fill="#0f3460" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
