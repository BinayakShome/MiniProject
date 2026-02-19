# 🎯 AURA — Autonomous Regime-Aware Portfolio Intelligence Engine

> 6th Semester Mini Project

---

## 👥 Team Members

| Name                   | Role                        |
| ---------------------- | --------------------------- |
| Debdyuti Chakraborty   | Quant Engine & Optimization |
| Binayak Shome          | Agent Architecture          |
| Atharva Pratap Singh   | Backtesting & Performance   |
| Aryan Sinha            | Monte Carlo & Risk          |
| Aryan Yadav            | Data Pipeline               |
| Deepjyoti Bhattacharya | Memory & State Management   |

**Project Guide:** Prof. Himanshu Ranjan

---

## 📌 What is AURA?

AURA is a fully autonomous, multi-agent portfolio management system for Indian ETFs. It combines quantitative optimization, market regime detection, user risk profiling, and Monte Carlo simulation to produce real-time, personalized portfolio allocations — with an optional AI-generated investment memo via a local LLM.

---

## 🏗️ Architecture

```
main.py
├── data/
│   └── data_loader.py          — Fetches OHLCV data via yfinance
├── quant/
│   ├── metrics.py              — Log returns, Sharpe, covariance
│   ├── optimization.py         — Max-Sharpe & Min-Vol (SciPy SLSQP)
│   ├── monte_carlo.py          — Bootstrap Monte Carlo simulation
│   └── performance.py          — CAGR, Max Drawdown, Calmar Ratio
├── agents/
│   ├── risk_profile_agent.py   — User risk scoring & equity cap
│   ├── market_regime_agent.py  — Bull / Bear / Sideways / High-Vol detection
│   ├── decision_agent.py       — Turnover-based rebalance trigger
│   └── explanation_agent.py    — LLM investment memo (Ollama / phi3:mini)
└── memory/
    └── state_manager.py        — JSON-based persistent portfolio state
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
pip install numpy scipy pandas matplotlib yfinance requests
```

### (Optional) AI Memo via Ollama

```bash
# Install Ollama from https://ollama.com
ollama serve
ollama pull phi3:mini
```

> If Ollama is not running, AURA still completes fully — the memo section prints a plain-text summary instead of crashing.

---

## ▶️ Running the Program

```bash
cd MiniProject-main
python main.py
```

You will be prompted for:

```
Enter your age: 30
Investment horizon (years): 10
Risk tolerance (Low/Medium/High): High
```

### Example Output

```
=Downloading data...
[*********************100%***********************]  4 of 4 completed
Download complete
Assets downloaded: ['GOLDBEES.NS', 'LIQUIDBEES.NS', 'MID150BEES.NS', 'NIF100BEES.NS']
Shape: (1626, 4)
Enter your age: 30
Investment horizon (years): 10
Risk tolerance (Low/Medium/High): High

==============================
Autonomous Portfolio Decision
==============================
Risk Score: 0.67
Market Regime: Sideways Market
Turnover: 0.014722767723227698
Rebalance Recommended: True

Final Allocation:
  NIF100BEES.NS: 0.053
  MID150BEES.NS: 0.3258
  GOLDBEES.NS: 0.6212
  LIQUIDBEES.NS: 0.0

Performance Metrics:
  Return: 0.212
  Volatility: 0.1276
  Sharpe: 1.1914
  CAGR: 0.2357
  Max Drawdown: -0.1933
  Calmar Ratio: 1.2193

Monte Carlo Projection (10-Year Forward):
  Expected Final Value: 9.061
  5% VaR: 4.278
  5% CVaR (Expected Shortfall): 3.623
  Probability of Loss: 0.0
```

#### Example Memo Using Ollama
```
Generating Investment Memo...


===== AI Portfolio Memo =====


**Institutional Investment Memo: ETF Portfolio Allocation & Outlook in a Sideways Market Environment**

Date: [Insert Date]

To, From: Institution Managed by DisciplinedPortfolios Ltd.

Subject: Ten-Year Horizon and Diversified Equity Exposure Through Indian Large-cap ETFs (NIF100BEES), Mid-cap ETFs (MID150BEES), Gold ETF (GOLDBEES), & Liquid ETF(LIQUIDBEES)

Portfolio Allocation Rationale:
Our tenure in a sideways market necessitates an allocation that seeks growth while mitigating risk. The portfolio comprises of the following allocations - NIF100BEES (Large-cap equity ETF): 5.3% with MID150BEES: 32.58%, GOLDBEES (Gold ETF) at a weightage of 62.12%, and LIQUIDBEES (Liquid ETF): zero allocation to capitalize on the current market conditions without immediate need for liquidity conversion, which provides us with flexibility in strategy adaptation as needed over our investment horizon.

Risk-Adjusted Performance Metrics:
1) Sharpe Ratio stands at an appreciable value of 1.1914 indicating the excess return per unit of volatility risk taken, which showcases relative favorability in comparison to other benchmarks given our investment objectives and constraints;
2) Calmar Ratio (return to drawdown ratio), a key metric for short-term liquidity analysis is at 1.2193 further corroborating the robustness of this portfolio against significant losses relative to its inflow periods within the past years, while accounting only up until our last fiscal quarter;
3) Our maximum drawdown stands modestly lowered by -0.1933 or 19.33%, providing us a clear perspective of market risk absorption capacity and emphasizing that such levels are within an acceptable range for long-term investors in sideways markets with the given exposures;
4) The Expected Final Value stands at a wealth multiple benchmarked by our projections, which is 9.061 times without any currency considerations over this tenure horizon.

Monte Carlo Outlook:
Our Monte Carlo simulations provide an in-depth quantitative forecast of expected returns and associated uncertainties for the portfolio across various market scenarios to anticipate outcomes effectively;
1) We predict our Expected Final Value at a multiple benchmarked by model projections, which stands firmly as 9.061 times without currency implications over this tenure horizon - an indicator of positive wealth accumulation when we align with historical patterns and current macro trends;
2) Our VaR estimate shows potential downside loss within the next year at a value benchmarked by our model projections to be 4.278 times without currency implications, which provides us insight into possible losses given standard market conditions over one-year periods from now and helps in contingency planning;
3) The CVaR or expected shortfall is projected at a potential loss benchmark of our model simulations as being equivalent to 3.623 times without currency implications, which represents an essential aspect of downside risk assessment where we evaluate the average outcome beyond VaR and serve in strategic decision-making;
4) The Probability of Loss within a year stands at zero percent by our model projections indicating no expected loss occurrences over this timeframe under current market conditions, which reinforces confidence while planning for long haul investments.

Balanced Forward-Looking Risk Commentary:
Considering the projected metrics and allocation rationale in a sideways to slightly upwards moving Indian equity contextualized within our tenure horizon of ten years, we are positioned optimally for consistent growth with diversification across capitalization bands. Our strategic mix accounts not only for potential upside but also ensures relative immunity against market downfalls given the current economic trajectory and historical data patterns in India’s equity landscape which typically reflect a cyclical nature where these sideways trends are commonplace;

We anticipate leveraging our allocation strategy to optimize returns while maintaining necessary liquidity, as well capitalize on growth opportunities within large-cap ETFs with lesser mid and small caps due to the general risk tolerance of institutional clients we manage. Our investment approach remains proactive in monitoring market indicators that could impact volatility or expected return metrics while remaining adaptive should a deviation from our anticipated outcomes occur;

The Monte Carlo simulations provide us with an essential toolset for ongoing quantitative risk assessment and strategic recalibration. We will maintain rigorous oversight of market regimes, investor sentiments towards equity markets in India particularly within sectors predominantly represented by our selected ETFs, inflation expectations which may impact interest rates thereby influencing the cost of capital for future allocations;

Our portfolio’s calibration and risk management processes will continue to operate under disciplined monitoring with periodic reviews ensuring alignment between investment strategies and overarching financial goals in a manner that reflects our client's unique institutional mandates.

Respectfully, the Disciplined Institutional Portfolio Manager Team at DisciplinedPortfolios Ltd., Mumbai
```

---

## 🧠 How It Works

### 1. Data Loading

Downloads daily Close prices for 4 Indian ETFs from Yahoo Finance (NSE):

- `NIF100BEES.NS` — Nifty 100 (large-cap equity)
- `MID150BEES.NS` — Nifty Midcap 150 (mid-cap equity)
- `GOLDBEES.NS` — Gold ETF
- `LIQUIDBEES.NS` — Liquid / overnight ETF

### 2. Quantitative Optimization

Computes annualised log returns and covariance matrix, then runs **Max Sharpe Ratio** optimization (SciPy SLSQP) to find the efficient frontier allocation.

### 3. Risk Profile Agent

Scores the user on a `[0, 1]` scale from age, investment horizon, and risk tolerance. Caps equity exposure at `0.8 × risk_score` and redistributes excess to liquid assets.

**Input validation (v2):** Raises `ValueError` on negative age, negative horizon, or unrecognised tolerance string.

### 4. Market Regime Detection

Classifies the current market into one of four regimes based on rolling volatility and mean return, then tilts the allocation accordingly:

| Regime          | Equity Adjustment |
| --------------- | ----------------- |
| Bull Market     | +10% equity       |
| High Volatility | −20% equity       |
| Bear Market     | −30% equity       |
| Sideways Market | No change         |

### 5. Monte Carlo Simulation

Bootstraps 5,000 paths over the user's investment horizon. Reports:

- **Expected Final Value** — mean wealth multiple
- **5% VaR** — worst-case at 5th percentile
- **5% CVaR** — expected shortfall below VaR
- **Probability of Loss** — fraction of paths ending below 1×

### 6. Decision Agent

Compares current weights to the last saved allocation. Recommends rebalancing if turnover exceeds 10%.

### 7. State Manager

Saves the final allocation, regime, performance metrics, and risk score to `memory/portfolio_state.json` for use on the next run.

### 8. AI Investment Memo (optional)

Sends all metrics to a local `phi3:mini` model via Ollama to generate an institutional-style investment memo. Degrades gracefully if Ollama is offline.

---

## 🔬 Stress Testing

A full stress test suite is included (`stress_test.py`) covering 67 tests across all modules:

```bash
python stress_test.py
```

Tests cover zero/negative/NaN/Inf prices, degenerate weight vectors, singular covariance matrices, zero-simulation Monte Carlo, CAGR on zero-start portfolios, missing ticker keys in agents, corrupted JSON state, and more.

**Latest results:** 56 passed · 0 bugs · 11 design warnings

---

## 🐛 Bug Fixes (v2)

| Module                          | Bug                                                      | Fix                                                        |
| ------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| `quant/metrics.py`              | Sharpe = `-inf` when portfolio vol = 0                   | Returns `0.0` when vol is zero                             |
| `quant/monte_carlo.py`          | `IndexError` crash when `simulations=0` or `years=0`     | Raises `ValueError` with clear message                     |
| `quant/monte_carlo.py`          | `exp()` overflow on extreme returns                      | Cumulative returns clipped to `[-700, 700]` before `exp()` |
| `quant/performance.py`          | `CAGR = inf` when start value is 0                       | Raises `ValueError`                                        |
| `quant/performance.py`          | `max_drawdown = NaN` on all-zero portfolio               | Returns `0.0` safely                                       |
| `quant/performance.py`          | `annual_vol = NaN` for single observation                | Returns `0.0` when `len < 2`                               |
| `agents/risk_profile_agent.py`  | Negative age/horizon accepted silently                   | Raises `ValueError` on invalid inputs                      |
| `agents/risk_profile_agent.py`  | Unknown tolerance silently treated as medium             | Raises `ValueError`                                        |
| `agents/risk_profile_agent.py`  | `KeyError` when `LIQUIDBEES.NS` absent                   | Redistributes excess to any present non-equity asset       |
| `agents/market_regime_agent.py` | `KeyError` on missing ticker in `adjust_for_regime`      | Only adjusts tickers present in the portfolio              |
| `agents/market_regime_agent.py` | NaN comparisons in `detect_regime` silently fall through | Guards with `np.isfinite()` before comparisons             |
| `memory/state_manager.py`       | `numpy.float64` not JSON-serialisable                    | Added `_NumpyEncoder` custom JSON encoder                  |
| `agents/explanation_agent.py`   | Full crash when Ollama is offline                        | `try/except` with graceful fallback message                |

---

## 📁 Project Structure

```
MiniProject-main/
├── main.py
├── README.md
├── stress_test.py
├── data/
│   ├── __init__.py
│   └── data_loader.py
├── quant/
│   ├── metrics.py
│   ├── optimization.py
│   ├── monte_carlo.py
│   └── performance.py
├── agents/
│   ├── risk_profile_agent.py
│   ├── market_regime_agent.py
│   ├── decision_agent.py
│   └── explanation_agent.py
└── memory/
    ├── state_manager.py
    └── portfolio_state.json
```

---

## 📄 License

See [LICENSE](LICENSE)
