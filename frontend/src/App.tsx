import axios from 'axios'
import { useEffect, useState } from 'react'
import PDCalculator from './components/PDCalculator'
import StressDashboard from './components/StressDashboard'
import ModelMetricsView from './components/ModelMetrics'
import SessionLogin from './components/SessionLogin'
import { api } from './api'

type Page = 'pd' | 'stress' | 'metrics'

function isAuthError(error: unknown): boolean {
  return axios.isAxiosError(error) && error.response?.status === 401
}

export default function App() {
  const [page, setPage] = useState<Page>('pd')
  const [authenticated, setAuthenticated] = useState<boolean | null>(null)

  useEffect(() => {
    let cancelled = false
    const bootstrap = async () => {
      try {
        await api.metrics()
        if (!cancelled) {
          setAuthenticated(true)
        }
      } catch (error) {
        if (!cancelled) {
          if (isAuthError(error)) {
            setAuthenticated(false)
          } else {
            setAuthenticated(true)
          }
        }
      }
    }
    bootstrap()
    return () => {
      cancelled = true
    }
  }, [])

  if (authenticated === null) {
    return <div className="app"><main className="main">Loading…</main></div>
  }

  return (
    <div className="app">
      {!authenticated && <SessionLogin onSuccess={() => setAuthenticated(true)} />}
      <nav className="nav">
        <h1>Credit Risk PD Engine</h1>
        <button className={page === 'pd' ? 'active' : ''} onClick={() => setPage('pd')}>PD Calculator</button>
        <button className={page === 'stress' ? 'active' : ''} onClick={() => setPage('stress')}>Stress Test</button>
        <button className={page === 'metrics' ? 'active' : ''} onClick={() => setPage('metrics')}>Model Metrics</button>
      </nav>
      <main className="main">
        {page === 'pd' && <PDCalculator />}
        {page === 'stress' && <StressDashboard />}
        {page === 'metrics' && <ModelMetricsView />}
      </main>
    </div>
  )
}
