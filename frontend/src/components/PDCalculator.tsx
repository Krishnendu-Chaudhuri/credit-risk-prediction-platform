import { useState } from 'react'
import { api, LoanInput, Prediction } from '../api'
import { bandStyle, formatCurrency, formatPd } from '../displayContract'

const defaultLoan: LoanInput = {
  person_age: 35,
  person_income: 60000,
  person_home_ownership: 'RENT',
  person_emp_length: 5,
  loan_intent: 'PERSONAL',
  loan_grade: 'C',
  loan_amnt: 10000,
  loan_int_rate: 12,
  loan_percent_income: 0.3,
  cb_person_default_on_file: 'N',
  cb_person_cred_hist_length: 3,
}

export default function PDCalculator() {
  const [loan, setLoan] = useState<LoanInput>(defaultLoan)
  const [model, setModel] = useState('xgb')
  const [result, setResult] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const update = (key: keyof LoanInput, value: string | number) => {
    setLoan((prev) => ({ ...prev, [key]: value }))
  }

  const handleSubmit = async () => {
    setLoading(true)
    setError('')
    try {
      const { data } = await api.predict(loan, model)
      setResult(data.predictions[0])
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const band = result ? bandStyle(result.risk_band) : null

  return (
    <div>
      <div className="card">
        <h2>Loan PD Calculator</h2>
        <div className="grid">
          <div>
            <label>Person Age</label>
            <input type="number" value={loan.person_age} onChange={(e) => update('person_age', +e.target.value)} />
            <label>Person Income</label>
            <input type="number" value={loan.person_income} onChange={(e) => update('person_income', +e.target.value)} />
            <label>Home Ownership</label>
            <select value={loan.person_home_ownership} onChange={(e) => update('person_home_ownership', e.target.value)}>
              <option>RENT</option><option>OWN</option><option>MORTGAGE</option>
            </select>
            <label>Employment Length</label>
            <input type="number" value={loan.person_emp_length} onChange={(e) => update('person_emp_length', +e.target.value)} />
          </div>
          <div>
            <label>Loan Intent</label>
            <select value={loan.loan_intent} onChange={(e) => update('loan_intent', e.target.value)}>
              <option>PERSONAL</option><option>EDUCATION</option><option>MEDICAL</option>
              <option>VENTURE</option><option>HOMEIMPROVEMENT</option><option>DEBTCONSOLIDATION</option>
            </select>
            <label>Loan Grade</label>
            <select value={loan.loan_grade} onChange={(e) => update('loan_grade', e.target.value)}>
              {['A','B','C','D','E','F','G'].map((g) => <option key={g}>{g}</option>)}
            </select>
            <label>Loan Amount</label>
            <input type="number" value={loan.loan_amnt} onChange={(e) => update('loan_amnt', +e.target.value)} />
            <label>Interest Rate (%)</label>
            <input type="number" value={loan.loan_int_rate} onChange={(e) => update('loan_int_rate', +e.target.value)} />
          </div>
          <div>
            <label>Loan % Income</label>
            <input type="number" step="0.01" value={loan.loan_percent_income} onChange={(e) => update('loan_percent_income', +e.target.value)} />
            <label>Prior Default</label>
            <select value={loan.cb_person_default_on_file} onChange={(e) => update('cb_person_default_on_file', e.target.value)}>
              <option>Y</option><option>N</option>
            </select>
            <label>Credit History Length</label>
            <input type="number" value={loan.cb_person_cred_hist_length} onChange={(e) => update('cb_person_cred_hist_length', +e.target.value)} />
            <label>Model</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="xgb">XGBoost</option>
              <option value="lr">Logistic Regression</option>
            </select>
          </div>
        </div>
        <button className="primary" onClick={handleSubmit} disabled={loading}>
          {loading ? 'Calculating...' : 'Calculate PD'}
        </button>
        {error && <p className="error">{error}</p>}
      </div>

      {result && band && (
        <div className="card">
          <h3>Results</h3>
          <div className="metrics-row">
            <div className="metric"><div className="value">{formatPd(result.pd)}</div><div className="label">PD</div></div>
            <div className="metric"><div className="value">{result.risk_score}</div><div className="label">Risk Score</div></div>
            <div className="metric">
              <div className="value" style={{ color: band.color }}>{band.label}</div>
              <div className="label">Risk Band</div>
            </div>
            <div className="metric"><div className="value">{formatCurrency(result.ecl)}</div><div className="label">ECL (IFRS 9)</div></div>
            <div className="metric"><div className="value">Stage {result.ifrs9_stage}</div><div className="label">IFRS 9 Stage</div></div>
            <div className="metric"><div className="value">{formatCurrency(result.expected_loss)}</div><div className="label">EL (Basel)</div></div>
          </div>
          {result.reason_codes && result.reason_codes.length > 0 && (
            <p><strong>Reason codes:</strong> {result.reason_codes.join(', ')}</p>
          )}
        </div>
      )}
    </div>
  )
}
