import { FormEvent, useState } from 'react'
import { establishSession } from '../api'

interface SessionLoginProps {
  onSuccess: () => void
}

export default function SessionLogin({ onSuccess }: SessionLoginProps) {
  const [apiKey, setApiKey] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      await establishSession(apiKey)
      onSuccess()
    } catch {
      setError('Invalid API key. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="session-login-overlay">
      <form className="session-login-modal" onSubmit={handleSubmit}>
        <h2>Sign in</h2>
        <p>Enter your API key to access the dashboard.</p>
        <label htmlFor="api-key">API Key</label>
        <input
          id="api-key"
          type="password"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          autoComplete="current-password"
          required
        />
        {error && <p className="error">{error}</p>}
        <button type="submit" disabled={loading}>
          {loading ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
    </div>
  )
}
