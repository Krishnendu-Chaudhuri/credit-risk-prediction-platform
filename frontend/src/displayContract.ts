import contractJson from '../../configs/display_contract.json'

export interface RiskBandStyle {
  color: string
  label: string
  pd_max: number
}

export interface DisplayContract {
  risk_bands: Record<string, RiskBandStyle>
  format: {
    pd: string
    currency: string
  }
}

export const displayContract = contractJson as DisplayContract

export function bandStyle(riskBand: string): RiskBandStyle {
  return displayContract.risk_bands[riskBand] ?? {
    color: '#666666',
    label: riskBand,
    pd_max: 1,
  }
}

export function formatPd(pd: number): string {
  return `${(pd * 100).toFixed(2)}%`
}

export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount)
}
