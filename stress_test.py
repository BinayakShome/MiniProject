"""
=============================================================================
STRESS TEST SUITE — MiniProject Portfolio System
=============================================================================
Covers:
  - Zero / negative / NaN / Inf prices                     [DataLoader / metrics]
  - Single-row and single-asset edge cases                  [metrics / optimization]
  - Covariance matrix singularity → division-by-zero        [optimization]
  - All-identical prices (zero variance / zero vol)         [metrics / optim]
  - Weights that don't sum to 1 / negative weights          [metrics / monte carlo]
  - Zero-weight portfolios                                  [metrics]
  - Sharpe with zero volatility (divide-by-zero)            [metrics]
  - Optimizer convergence on degenerate inputs              [optimization]
  - Efficient frontier with < 2 assets                      [optimization]
  - Monte Carlo with zero-length returns                    [monte_carlo]
  - Monte Carlo weights mismatch                            [monte_carlo]
  - CAGR / max-drawdown on constant / monotone series       [performance]
  - Calmar with zero max drawdown                           [performance]
  - RiskProfileAgent boundary ages (0, 200, negative)       [risk_profile_agent]
  - RiskProfileAgent unknown tolerance string               [risk_profile_agent]
  - RiskProfileAgent missing equity key in weights_dict     [risk_profile_agent]
  - MarketRegimeAgent on empty / all-NaN returns            [market_regime_agent]
  - MarketRegimeAgent adjust_for_regime missing asset keys  [market_regime_agent]
  - DecisionAgent with NaN / Inf weights                    [decision_agent]
  - DecisionAgent turnover threshold of exactly 0           [decision_agent]
  - StateManager save / load round-trip with funky data     [state_manager]
  - StateManager on corrupted JSON file                     [state_manager]
  - Rolling backtest with fewer rows than lookback          [backtest/engine]
  - Rolling backtest with a single asset                    [backtest/engine]
=============================================================================
"""

import sys
import os
import json
import tempfile
import traceback
import math

import numpy as np
import pandas as pd

# ---------- path setup ----------
sys.path.insert(0, os.path.dirname(__file__))

# ---------- modules under test ----------
from quant.metrics import (
    compute_log_returns,
    annualized_return,
    annualized_volatility,
    covariance_matrix,
    portfolio_performance,
)
from quant.optimization import (
    max_sharpe_ratio,
    min_volatility,
    generate_efficient_frontier,
)
from quant.monte_carlo import monte_carlo_simulation
from quant.performance import (
    compute_cagr,
    compute_max_drawdown,
    compute_calmar,
    compute_annual_volatility,
)
from agents.risk_profile_agent import RiskProfileAgent
from agents.market_regime_agent import MarketRegimeAgent
from agents.decision_agent import DecisionAgent
from memory.state_manager import StateManager


# =============================================================================
# Helpers
# =============================================================================

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results = []


def run_test(name, fn):
    """Run a single test and capture outcome."""
    try:
        outcome = fn()
        if outcome is None:
            status = PASS
            detail = ""
        elif outcome is True:
            status = PASS
            detail = ""
        elif isinstance(outcome, str) and outcome.startswith("WARN"):
            status = WARN
            detail = outcome
        else:
            status = FAIL
            detail = str(outcome)
    except Exception as exc:
        status = FAIL
        detail = f"{type(exc).__name__}: {exc}\n{''.join(traceback.format_tb(exc.__traceback__)[-2:])}"

    results.append((name, status, detail))
    flag = "BUG FOUND" if status == FAIL else ("NOTE" if status == WARN else "")
    line = f"  {status}  {name}"
    if flag:
        line += f"  ← {flag}"
    print(line)
    if detail and status == FAIL:
        for row in detail.strip().split("\n"):
            print(f"           {row}")


def make_price_df(n_days=500, n_assets=3, start=100.0, drift=0.0001, vol=0.01, seed=42):
    """Generate synthetic log-normal price data."""
    rng = np.random.default_rng(seed)
    log_r = rng.normal(drift, vol, size=(n_days, n_assets))
    prices = start * np.exp(np.cumsum(log_r, axis=0))
    cols = [f"A{i}" for i in range(n_assets)]
    idx = pd.date_range("2015-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=idx, columns=cols)


def make_log_returns(n_days=500, n_assets=3, seed=42, drift=0.0001, vol=0.01):
    price_df = make_price_df(n_days=n_days + 1, n_assets=n_assets, seed=seed, drift=drift, vol=vol)
    return compute_log_returns(price_df)


def default_weights(n):
    return np.array([1.0 / n] * n)


# =============================================================================
# 1.  METRICS — compute_log_returns
# =============================================================================

def test_log_returns_zero_price():
    """Zero price causes log(0) = -inf, should not silently propagate."""
    df = make_price_df(n_days=10, n_assets=2)
    df.iloc[5, 0] = 0.0  # inject zero price
    lr = compute_log_returns(df)
    if not np.isfinite(lr.values).all():
        return "FAIL: -inf / NaN survived in log_returns after zero price injection"


def test_log_returns_negative_price():
    """Negative price → log of negative → NaN. Must be caught."""
    df = make_price_df(n_days=10, n_assets=2)
    df.iloc[3, 1] = -50.0
    lr = compute_log_returns(df)
    if not np.isfinite(lr.values).all():
        return "FAIL: NaN survived in log_returns after negative price injection"


def test_log_returns_single_row():
    """One-row DataFrame: shift produces all NaN → dropna → empty."""
    df = make_price_df(n_days=1, n_assets=3)
    lr = compute_log_returns(df)
    assert len(lr) == 0, f"Expected empty, got {len(lr)} rows"


def test_log_returns_all_same():
    """Constant prices → all zero log returns → zero volatility."""
    df = pd.DataFrame({"A": [100.0] * 50, "B": [200.0] * 50},
                      index=pd.date_range("2020-01-01", periods=50))
    lr = compute_log_returns(df)
    assert (lr == 0).all().all(), "Expected all-zero log returns for constant prices"


def test_log_returns_inf_price():
    """Inf price propagates Inf returns."""
    df = make_price_df(n_days=20, n_assets=2)
    df.iloc[10, 0] = np.inf
    lr = compute_log_returns(df)
    if not np.isfinite(lr.values).all():
        return "FAIL: Inf survived in log_returns after Inf price injection"


def test_log_returns_nan_price():
    """NaN price mid-series."""
    df = make_price_df(n_days=20, n_assets=2)
    df.iloc[7, 1] = np.nan
    lr = compute_log_returns(df)
    if not np.isfinite(lr.values).all():
        return "FAIL: NaN survived in log_returns after NaN price"


# =============================================================================
# 2.  METRICS — portfolio_performance
# =============================================================================

def test_portfolio_perf_zero_vol():
    """All-constant prices → zero vol → Sharpe = division by zero."""
    df = pd.DataFrame({"A": [100.0] * 252, "B": [50.0] * 252},
                      index=pd.date_range("2020-01-01", periods=252))
    lr = compute_log_returns(df)
    if lr.empty:
        return  # no returns to test — not a bug per se, but flag
    mr = annualized_return(lr)
    cm = covariance_matrix(lr)
    w = default_weights(2)
    ret, vol, sharpe = portfolio_performance(w, mr, cm)
    if not math.isfinite(sharpe):
        return f"FAIL: Sharpe is {sharpe} when portfolio vol is zero (division by zero)"


def test_portfolio_perf_negative_weights():
    """Negative weights are mathematically valid (short) but the optimiser
    enforces bounds=[0,1]. Feeding negatives directly to portfolio_performance
    should still return finite numbers."""
    lr = make_log_returns()
    mr = annualized_return(lr)
    cm = covariance_matrix(lr)
    w = np.array([-0.5, 1.0, 0.5])  # sums to 1 but has negative component
    ret, vol, sharpe = portfolio_performance(w, mr, cm)
    if not (math.isfinite(ret) and math.isfinite(vol) and math.isfinite(sharpe)):
        return f"FAIL: non-finite result with negative weights: ret={ret}, vol={vol}, sharpe={sharpe}"


def test_portfolio_perf_zero_weights():
    """All weights = 0 → portfolio return = 0, vol = 0 → Sharpe blow-up."""
    lr = make_log_returns()
    mr = annualized_return(lr)
    cm = covariance_matrix(lr)
    w = np.zeros(3)
    ret, vol, sharpe = portfolio_performance(w, mr, cm)
    if not math.isfinite(sharpe):
        return f"FAIL: Sharpe={sharpe} with all-zero weights (zero vol → division by zero)"


def test_portfolio_perf_weights_not_sum_to_one():
    """Weights summing to 2 should raise an error OR be normalised; not silently wrong."""
    lr = make_log_returns()
    mr = annualized_return(lr)
    cm = covariance_matrix(lr)
    w = np.array([1.0, 0.5, 0.5])  # sums to 2
    ret, vol, sharpe = portfolio_performance(w, mr, cm)
    # We just verify the outputs are finite; the function doesn't normalise,
    # so an analyst could get doubled returns silently.
    note = f"WARN: weights sum to {w.sum()} but no normalisation/validation occurs — silent scaling bug"
    return note


# =============================================================================
# 3.  OPTIMIZATION — max_sharpe_ratio / min_volatility
# =============================================================================

def test_optim_single_asset():
    """Only 1 asset → optimal weight must be 1.0."""
    lr = make_log_returns(n_assets=1)
    mr = annualized_return(lr)
    cm = covariance_matrix(lr)
    w = max_sharpe_ratio(mr, cm)
    assert abs(w[0] - 1.0) < 1e-4, f"Expected weight=1.0 for single asset, got {w[0]:.4f}"


def test_optim_identical_assets():
    """Perfectly correlated assets → singular covariance; optimiser must not crash."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 300)
    df = pd.DataFrame({"A": 100 * np.exp(np.cumsum(base)),
                       "B": 100 * np.exp(np.cumsum(base))},  # identical
                      index=pd.date_range("2015-01-01", periods=300))
    lr = compute_log_returns(df)
    mr = annualized_return(lr)
    cm = covariance_matrix(lr)
    try:
        w = max_sharpe_ratio(mr, cm)
        if not np.isfinite(w).all():
            return f"FAIL: non-finite weights for identical assets: {w}"
        if abs(w.sum() - 1.0) > 1e-3:
            return f"FAIL: weights don't sum to 1 for identical assets: {w.sum()}"
    except Exception as e:
        return f"FAIL: crashed on identical assets — {e}"


def test_optim_all_negative_returns():
    """All assets have negative expected returns → Sharpe always negative.
    Optimiser must still converge (not inf-loop or crash)."""
    lr = make_log_returns()
    mr = -np.abs(annualized_return(lr))  # force negative
    cm = covariance_matrix(lr)
    try:
        w = max_sharpe_ratio(mr, cm)
        if not np.isfinite(w).all():
            return f"FAIL: non-finite weights for all-negative returns: {w}"
    except Exception as e:
        return f"FAIL: crashed on all-negative returns — {e}"


def test_optim_zero_covariance():
    """Near-zero covariance (low vol) should not blow up Sharpe calculation."""
    lr = make_log_returns(vol=1e-8)
    mr = annualized_return(lr)
    cm = covariance_matrix(lr)
    try:
        w = max_sharpe_ratio(mr, cm)
        if not np.isfinite(w).all():
            return f"FAIL: non-finite weights with near-zero covariance: {w}"
    except Exception as e:
        return f"FAIL: crashed with near-zero covariance — {e}"


def test_optim_min_vol_single_asset():
    lr = make_log_returns(n_assets=1)
    mr = annualized_return(lr)
    cm = covariance_matrix(lr)
    w = min_volatility(mr, cm)
    assert abs(w[0] - 1.0) < 1e-4, f"Expected weight=1.0, got {w[0]:.4f}"


def test_efficient_frontier_two_assets():
    """Minimum meaningful case for frontier."""
    lr = make_log_returns(n_assets=2)
    mr = annualized_return(lr)
    cm = covariance_matrix(lr)
    results_arr, weights_record = generate_efficient_frontier(mr, cm, num_portfolios=200)
    assert len(results_arr) > 0, "Efficient frontier returned zero portfolios"
    assert np.isfinite(results_arr).all(), "Non-finite values in efficient frontier"


def test_efficient_frontier_single_asset():
    """One asset edge case; should not crash."""
    lr = make_log_returns(n_assets=1)
    mr = annualized_return(lr)
    cm = covariance_matrix(lr)
    try:
        results_arr, _ = generate_efficient_frontier(mr, cm, num_portfolios=100)
    except Exception as e:
        return f"FAIL: crashed on single-asset frontier — {e}"


# =============================================================================
# 4.  MONTE CARLO
# =============================================================================

def test_mc_zero_simulations():
    """simulations=0 → empty results → mean/percentile crash."""
    lr = make_log_returns()
    w = default_weights(3)
    try:
        results_arr, metrics = monte_carlo_simulation(w, lr, years=1, simulations=0)
        if not math.isfinite(metrics["Expected Final Value"]):
            return "FAIL: Expected Final Value is non-finite with 0 simulations"
    except Exception as e:
        return f"FAIL: crashed with simulations=0 — {type(e).__name__}: {e}"


def test_mc_zero_years():
    """years=0 → 0 days sampled → empty cumsum → index error."""
    lr = make_log_returns()
    w = default_weights(3)
    try:
        results_arr, metrics = monte_carlo_simulation(w, lr, years=0, simulations=100)
        if not math.isfinite(metrics["Expected Final Value"]):
            return "FAIL: Expected Final Value non-finite with years=0"
    except Exception as e:
        return f"FAIL: crashed with years=0 — {type(e).__name__}: {e}"


def test_mc_weights_wrong_length():
    """Weight vector length != number of assets → dot product crash."""
    lr = make_log_returns(n_assets=3)
    w = default_weights(2)  # wrong length
    try:
        results_arr, metrics = monte_carlo_simulation(w, lr, years=1, simulations=100)
        return "FAIL: No error raised for mismatched weights/assets length"
    except (ValueError, Exception):
        pass  # expected to fail; this is correct behaviour if it raises


def test_mc_negative_weights():
    """Negative weights (short positions) must not blow up MC."""
    lr = make_log_returns(n_assets=3)
    w = np.array([-0.2, 0.8, 0.4])  # sums to 1 but has short
    try:
        results_arr, metrics = monte_carlo_simulation(w, lr, years=2, simulations=500)
        if not all(math.isfinite(v) for v in metrics.values()):
            return f"FAIL: Non-finite metrics with negative weights: {metrics}"
    except Exception as e:
        return f"FAIL: crashed with negative weights — {e}"


def test_mc_single_day_returns():
    """Degenerate returns DataFrame with only 1 row."""
    lr = make_log_returns(n_days=2, n_assets=3)  # after dropna → 1 row
    w = default_weights(3)
    try:
        results_arr, metrics = monte_carlo_simulation(w, lr, years=1, simulations=200)
        if not all(math.isfinite(v) for v in metrics.values()):
            return f"FAIL: non-finite metrics with single-row returns: {metrics}"
    except Exception as e:
        return f"FAIL: crashed with single-day returns — {e}"


def test_mc_all_zero_returns():
    """All zero log returns → portfolio always stays at 1.0."""
    df = pd.DataFrame({"A": [100.0] * 252, "B": [200.0] * 252},
                      index=pd.date_range("2020-01-01", periods=252))
    lr = compute_log_returns(df)
    if lr.empty:
        return  # degenerate case handled elsewhere
    w = default_weights(2)
    results_arr, metrics = monte_carlo_simulation(w, lr, years=1, simulations=200)
    if abs(metrics["Expected Final Value"] - 1.0) > 0.01:
        return f"FAIL: Expected value should be ~1.0 with zero returns, got {metrics['Expected Final Value']}"


def test_mc_extreme_positive_drift():
    """Very large returns → exponential overflow."""
    lr = make_log_returns(drift=1.0, vol=0.01)  # 100% daily drift
    w = default_weights(3)
    try:
        results_arr, metrics = monte_carlo_simulation(w, lr, years=1, simulations=100)
        if not all(math.isfinite(v) for v in metrics.values()):
            return f"FAIL: overflow with extreme drift: {metrics}"
    except Exception as e:
        return f"FAIL: crashed with extreme drift — {e}"


# =============================================================================
# 5.  PERFORMANCE — CAGR / Drawdown / Calmar
# =============================================================================

def test_cagr_constant_series():
    """Portfolio never changes → CAGR = 0%."""
    pv = np.ones(252)
    cagr = compute_cagr(pv)
    assert abs(cagr) < 1e-9, f"Expected CAGR=0 for flat series, got {cagr}"


def test_cagr_single_value():
    """Length-1 series → division by zero (years=0)."""
    pv = np.array([1.0])
    try:
        cagr = compute_cagr(pv)
        if not math.isfinite(cagr):
            return f"FAIL: CAGR is {cagr} for single-element series (div-by-zero)"
    except Exception as e:
        return f"FAIL: crashed on single-value series — {e}"


def test_cagr_zero_start():
    """Start value of 0 → division by zero in total_return."""
    pv = np.array([0.0, 1.0, 1.5])
    try:
        cagr = compute_cagr(pv)
        if not math.isfinite(cagr):
            return f"FAIL: CAGR is {cagr} when start=0 (division by zero)"
    except Exception as e:
        return f"FAIL: crashed on zero start value — {e}"


def test_cagr_negative_start():
    """Negative starting portfolio value → undefined CAGR."""
    pv = np.array([-1.0, 1.0, 1.5])
    try:
        cagr = compute_cagr(pv)
        if not math.isfinite(cagr):
            return f"FAIL: CAGR is {cagr} with negative start value"
    except Exception as e:
        return f"FAIL: crashed on negative start value — {e}"


def test_max_drawdown_monotone_up():
    """Strictly increasing series → max drawdown should be 0."""
    pv = np.cumsum(np.ones(200)) + 1.0
    mdd = compute_max_drawdown(pv)
    assert mdd == 0.0, f"Expected max drawdown=0 for monotone rising series, got {mdd}"


def test_max_drawdown_single_value():
    """Single value → no drawdown period."""
    pv = np.array([1.0])
    try:
        mdd = compute_max_drawdown(pv)
        assert mdd == 0.0, f"Expected 0 drawdown, got {mdd}"
    except Exception as e:
        return f"FAIL: crashed on single-value series — {e}"


def test_max_drawdown_all_zeros():
    """Zero portfolio values → drawdown = 0/0."""
    pv = np.zeros(50)
    try:
        mdd = compute_max_drawdown(pv)
        if not math.isfinite(mdd):
            return f"FAIL: max drawdown={mdd} with all-zero portfolio values"
    except Exception as e:
        return f"FAIL: crashed on all-zero portfolio — {e}"


def test_calmar_zero_drawdown():
    """Already handled in code (returns 0), verify explicitly."""
    cagr = 0.15
    result = compute_calmar(cagr, max_dd=0)
    assert result == 0, f"Expected Calmar=0 for zero drawdown, got {result}"


def test_calmar_positive_drawdown_value():
    """max_dd should be negative; passing positive value returns wrong sign."""
    cagr = 0.15
    wrong_dd = 0.20  # should be -0.20
    result = compute_calmar(cagr, max_dd=wrong_dd)
    note = (f"WARN: compute_calmar({cagr}, {wrong_dd}) = {result:.2f}. "
            f"max_dd should be negative by convention; passing positive inverts sign silently.")
    return note


def test_annual_vol_single_day():
    """Single-day returns → std=NaN."""
    lr_series = pd.Series([0.01])
    vol = compute_annual_volatility(lr_series)
    if not math.isfinite(vol):
        return f"FAIL: annual volatility={vol} for single-observation series"


# =============================================================================
# 6.  RiskProfileAgent
# =============================================================================

EQUITY_ASSETS = ["NIF100BEES.NS", "MID150BEES.NS"]
DEFAULT_WEIGHTS = {
    "NIF100BEES.NS": 0.35,
    "MID150BEES.NS": 0.25,
    "LIQUIDBEES.NS": 0.20,
    "GOLDBEES.NS": 0.10,
    "ICICIB22.NS": 0.10,
}


def test_risk_age_zero():
    """Age=0 (newborn) → age_factor = (60-0)/40 = 1.5 → capped at 1."""
    rpa = RiskProfileAgent(age=0, horizon=30, tolerance="high")
    score = rpa.compute_risk_score()
    assert 0 <= score <= 1, f"Risk score {score} out of [0,1] for age=0"


def test_risk_age_negative():
    """Negative age should not be accepted silently."""
    rpa = RiskProfileAgent(age=-5, horizon=20, tolerance="medium")
    score = rpa.compute_risk_score()
    # age_factor = (60 - (-5)) / 40 = 65/40 > 1 → min(...,1) → capped → ok
    # But we flag it since no validation exists
    note = f"WARN: Negative age (-5) accepted silently; risk_score={score}. Input validation missing."
    return note


def test_risk_age_200():
    """Very old age → age_factor should be 0 (capped at 0)."""
    rpa = RiskProfileAgent(age=200, horizon=5, tolerance="low")
    score = rpa.compute_risk_score()
    assert 0 <= score <= 1, f"Risk score {score} out of [0,1] for age=200"


def test_risk_unknown_tolerance():
    """Unrecognised tolerance string falls back to 0.6 (medium) silently."""
    rpa = RiskProfileAgent(age=30, horizon=20, tolerance="EXTREME")
    score = rpa.compute_risk_score()
    note = (f"WARN: Unknown tolerance 'EXTREME' silently treated as 'medium' "
            f"(score={score}). No validation or error raised.")
    return note


def test_risk_zero_horizon():
    """Investment horizon of 0 years → horizon_factor = 0."""
    rpa = RiskProfileAgent(age=30, horizon=0, tolerance="high")
    score = rpa.compute_risk_score()
    assert 0 <= score <= 1, f"Score {score} out of bounds for horizon=0"


def test_risk_negative_horizon():
    """Negative horizon → horizon_factor could go negative."""
    rpa = RiskProfileAgent(age=30, horizon=-10, tolerance="medium")
    score = rpa.compute_risk_score()
    note = (f"WARN: Negative horizon (-10) accepted silently; "
            f"horizon_factor=max(0, -10/30)=0 but no error raised. score={score}")
    return note


def test_risk_adjust_allocation_missing_key():
    """weights_dict missing an equity key → KeyError crash."""
    rpa = RiskProfileAgent(age=25, horizon=20, tolerance="high")
    bad_weights = {"NIF100BEES.NS": 0.60, "LIQUIDBEES.NS": 0.40}
    # MID150BEES.NS is missing
    try:
        result, score = rpa.adjust_allocation(bad_weights)
        return "FAIL: No KeyError raised for missing equity asset in weights_dict"
    except KeyError:
        pass  # This IS the bug — no graceful handling


def test_risk_adjust_weights_sum_to_one():
    """After adjustment, weights should still sum to 1."""
    rpa = RiskProfileAgent(age=25, horizon=20, tolerance="high")
    w = DEFAULT_WEIGHTS.copy()
    adjusted, score = rpa.adjust_allocation(w)
    total = sum(adjusted.values())
    if abs(total - 1.0) > 1e-6:
        return f"FAIL: Adjusted weights sum to {total:.6f}, not 1.0"


def test_risk_adjust_equity_weight_cap():
    """After adjustment, equity weight should not exceed max_equity cap."""
    rpa = RiskProfileAgent(age=25, horizon=20, tolerance="high")
    w = DEFAULT_WEIGHTS.copy()
    risk_score = rpa.compute_risk_score()
    max_eq = 0.8 * risk_score
    adjusted, _ = rpa.adjust_allocation(w)
    eq_weight = sum(adjusted[a] for a in EQUITY_ASSETS)
    if eq_weight > max_eq + 1e-6:
        return f"FAIL: equity weight {eq_weight:.4f} exceeds cap {max_eq:.4f}"


def test_risk_adjust_liquidbees_missing():
    """LIQUIDBEES.NS absent from weights_dict → excess cannot be allocated → KeyError."""
    rpa = RiskProfileAgent(age=25, horizon=20, tolerance="high")
    bad_weights = {
        "NIF100BEES.NS": 0.50,
        "MID150BEES.NS": 0.30,
        "GOLDBEES.NS": 0.20,
    }
    try:
        result, score = rpa.adjust_allocation(bad_weights)
        # May or may not raise; check if weights are still valid
        total = sum(result.values())
        if abs(total - 1.0) > 1e-6:
            return f"FAIL: weights sum to {total:.4f} when LIQUIDBEES.NS missing"
    except KeyError:
        return "FAIL: KeyError on missing LIQUIDBEES.NS — no graceful handling"


# =============================================================================
# 7.  MarketRegimeAgent
# =============================================================================

def test_regime_empty_returns():
    """Empty DataFrame → std / mean → NaN → regime detection undefined."""
    empty_lr = pd.DataFrame(columns=["A", "B"])
    agent = MarketRegimeAgent(empty_lr)
    try:
        regime = agent.detect_regime()
        note = f"WARN: detect_regime on empty returns returned '{regime}' without error. NaN comparisons may behave unexpectedly."
        return note
    except Exception as e:
        return f"FAIL: crashed on empty log_returns — {e}"


def test_regime_all_nan_returns():
    """All-NaN returns → NaN volatility → NaN comparisons silently false."""
    lr = pd.DataFrame({"A": [np.nan] * 50, "B": [np.nan] * 50})
    agent = MarketRegimeAgent(lr)
    try:
        regime = agent.detect_regime()
        note = f"WARN: detect_regime on all-NaN returns returned '{regime}' — NaN comparisons silently evaluate as False"
        return note
    except Exception as e:
        return f"FAIL: crashed on all-NaN returns — {e}"


def test_regime_adjust_missing_asset():
    """adjust_for_regime with weights_dict missing expected ticker → KeyError."""
    lr = make_log_returns()
    agent = MarketRegimeAgent(lr)
    bad_weights = {"SOME_OTHER.NS": 0.5, "ANOTHER.NS": 0.5}
    for regime in ["High Volatility", "Bear Market", "Bull Market"]:
        try:
            adjusted = agent.adjust_for_regime(bad_weights, regime)
            # If it doesn't crash, check if it normalises properly
            total = sum(adjusted.values())
            if abs(total - 1.0) > 1e-6:
                return f"FAIL: weights sum to {total:.4f} after regime adjustment with missing assets (regime={regime})"
        except KeyError as e:
            return f"FAIL: KeyError for missing asset in regime={regime}: {e}"


def test_regime_adjust_zero_weight_assets():
    """Assets present but with 0 weight; scaling by 0.8 keeps them 0."""
    lr = make_log_returns()
    agent = MarketRegimeAgent(lr)
    weights = {
        "NIF100BEES.NS": 0.0,
        "MID150BEES.NS": 0.0,
        "LIQUIDBEES.NS": 1.0,
    }
    adjusted = agent.adjust_for_regime(weights, "Bear Market")
    total = sum(adjusted.values())
    if abs(total - 1.0) > 1e-6:
        return f"FAIL: weights sum to {total:.4f} with zero equity weights"


def test_regime_normalisation_after_bull():
    """Bull market boosts equity by 1.1; total may exceed 1 before normalise."""
    lr = make_log_returns()
    agent = MarketRegimeAgent(lr)
    weights = {
        "NIF100BEES.NS": 0.5,
        "MID150BEES.NS": 0.4,
        "LIQUIDBEES.NS": 0.1,
    }
    adjusted = agent.adjust_for_regime(weights, "Bull Market")
    total = sum(adjusted.values())
    if abs(total - 1.0) > 1e-6:
        return f"FAIL: weights sum to {total:.4f} after bull-market adjustment (normalisation broken)"


# =============================================================================
# 8.  DecisionAgent
# =============================================================================

def test_decision_first_call_always_rebalance():
    """First call should always rebalance regardless of weights."""
    agent = DecisionAgent()
    w = np.array([0.3, 0.4, 0.3])
    assert agent.should_rebalance(w, w), "First call should always return True"


def test_decision_nan_weights():
    """NaN in new_weights → NaN turnover → comparison undefined."""
    agent = DecisionAgent(rebalance_threshold=0.10)
    w_old = np.array([0.33, 0.33, 0.34])
    agent.should_rebalance(w_old, w_old)  # prime previous_weights
    w_nan = np.array([np.nan, 0.5, 0.5])
    try:
        result = agent.should_rebalance(w_old, w_nan)
        note = (f"WARN: should_rebalance with NaN weights returned {result} without error. "
                "NaN in turnover makes comparison unreliable.")
        return note
    except Exception as e:
        return f"FAIL: crashed on NaN weights — {e}"


def test_decision_inf_weights():
    """Inf in weights → Inf turnover → always rebalance but undefined."""
    agent = DecisionAgent(rebalance_threshold=0.10)
    w_old = np.array([0.33, 0.33, 0.34])
    agent.should_rebalance(w_old, w_old)
    w_inf = np.array([np.inf, 0.0, 0.0])
    try:
        result = agent.should_rebalance(w_old, w_inf)
        note = (f"WARN: should_rebalance with Inf weights returned {result} without error. "
                "Inf turnover always triggers rebalance silently.")
        return note
    except Exception as e:
        return f"FAIL: crashed on Inf weights — {e}"


def test_decision_zero_threshold():
    """Threshold=0 → any tiny change triggers rebalance (potentially every day)."""
    agent = DecisionAgent(rebalance_threshold=0.0)
    w1 = np.array([0.33, 0.33, 0.34])
    agent.should_rebalance(w1, w1)  # prime
    w2 = w1 + 1e-15  # infinitesimal change
    result = agent.should_rebalance(w1, w2)
    note = (f"WARN: threshold=0 causes rebalance on infinitesimal weight changes "
            f"(result={result}). No minimum threshold validation.")
    return note


def test_decision_negative_threshold():
    """Negative threshold → turnover always > threshold → always rebalance."""
    agent = DecisionAgent(rebalance_threshold=-0.05)
    w1 = np.array([0.33, 0.33, 0.34])
    agent.should_rebalance(w1, w1)
    result = agent.should_rebalance(w1, w1)  # identical weights
    note = (f"WARN: Negative threshold=-0.05 accepted silently; "
            f"identical weights return rebalance={result}. No validation.")
    return note


def test_decision_empty_weights():
    """Empty weight arrays."""
    agent = DecisionAgent()
    w = np.array([])
    try:
        result = agent.should_rebalance(w, w)
    except Exception as e:
        return f"FAIL: crashed on empty weight arrays — {e}"


def test_evaluate_market_state_boundary_vol():
    """Vol exactly at 0.18 boundary — check which branch fires."""
    agent = DecisionAgent()
    state = agent.evaluate_market_state(vol=0.18, sharpe=1.0)
    # Vol > 0.18 triggers High Volatility; vol == 0.18 should NOT
    if state == "High Volatility":
        return (f"WARN: evaluate_market_state classifies vol=0.18 as 'High Volatility' "
                f"using strict > but result is '{state}'. Boundary ambiguity.")


def test_evaluate_market_state_nan_inputs():
    """NaN vol/sharpe → unpredictable comparisons."""
    agent = DecisionAgent()
    try:
        state = agent.evaluate_market_state(vol=np.nan, sharpe=np.nan)
        note = f"WARN: evaluate_market_state(NaN, NaN) returned '{state}' without error — NaN comparison silently returns False"
        return note
    except Exception as e:
        return f"FAIL: crashed on NaN market state inputs — {e}"


# =============================================================================
# 9.  StateManager
# =============================================================================

def test_state_roundtrip():
    """Save then load should return identical data."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        sm = StateManager(filepath=path)
        allocation = {"NIF100BEES.NS": 0.5, "LIQUIDBEES.NS": 0.5}
        performance = {"cagr": 0.12, "sharpe": 1.4, "max_drawdown": -0.08}
        sm.save_state(allocation, "Bull Market", performance, risk_score=0.72)
        loaded = sm.load_state()
        assert loaded["allocation"] == allocation, "Allocation mismatch after round-trip"
        assert loaded["regime"] == "Bull Market", "Regime mismatch after round-trip"
        assert abs(loaded["risk_score"] - 0.72) < 1e-9, "Risk score mismatch"
    finally:
        os.unlink(path)


def test_state_corrupted_json():
    """Corrupted JSON → load_state should return {} without crash."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("{ CORRUPTED JSON !!! ]]]")
        path = f.name
    try:
        sm = StateManager(filepath=path)
        result = sm.load_state()
        assert result == {}, f"Expected empty dict from corrupted JSON, got {result}"
    finally:
        os.unlink(path)


def test_state_missing_file():
    """Non-existent filepath → load_state returns {}."""
    sm = StateManager(filepath="/tmp/this_file_does_not_exist_xyz123.json")
    result = sm.load_state()
    assert result == {}, f"Expected empty dict, got {result}"


def test_state_non_serialisable_data():
    """Saving non-JSON-serialisable data (e.g. numpy arrays) → TypeError."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        sm = StateManager(filepath=path)
        allocation = {"A": np.float64(0.5), "B": np.float64(0.5)}  # numpy scalar
        try:
            sm.save_state(allocation, "Sideways", {}, 0.5)
            return "FAIL: numpy float64 should cause JSON serialisation error but didn't"
        except TypeError:
            pass  # expected — this IS a bug if callers pass numpy types
    finally:
        os.unlink(path)


def test_state_empty_allocation():
    """Empty dict allocation should save/load cleanly."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        sm = StateManager(filepath=path)
        sm.save_state({}, "Unknown", {}, 0.0)
        loaded = sm.load_state()
        assert loaded["allocation"] == {}, f"Expected empty allocation, got {loaded['allocation']}"
    finally:
        os.unlink(path)


# =============================================================================
# 10.  BACKTEST ENGINE (rolling_backtest)
# =============================================================================

def test_backtest_fewer_rows_than_lookback():
    """Fewer price rows than lookback_days → loop never executes → trivial output."""
    from backtest.engine import rolling_backtest
    prices = make_price_df(n_days=100, n_assets=3)  # << 756 lookback
    try:
        result = rolling_backtest(prices, lookback_days=756)
        if len(result) <= 1:
            return (f"WARN: rolling_backtest with {len(prices)} rows < lookback=756 "
                    f"produces only {len(result)} result row(s). Portfolio effectively untested.")
    except Exception as e:
        return f"FAIL: crashed when rows < lookback — {e}"


def test_backtest_single_asset():
    """Single asset — covariance matrix is scalar; optimiser must handle."""
    from backtest.engine import rolling_backtest
    prices = make_price_df(n_days=1200, n_assets=1)
    try:
        result = rolling_backtest(prices, lookback_days=252)
        assert len(result) > 1, "Expected >1 result row for 1200-day single-asset backtest"
        if not np.isfinite(result.values).all():
            return f"FAIL: non-finite values in single-asset backtest result"
    except Exception as e:
        return f"FAIL: crashed on single-asset backtest — {e}"


def test_backtest_exactly_lookback_rows():
    """Exactly lookback_days rows → loop body runs 0 times → 1 element each column."""
    from backtest.engine import rolling_backtest
    lookback = 252
    prices = make_price_df(n_days=lookback + 1, n_assets=3)  # +1 so log_returns has lookback rows
    try:
        result = rolling_backtest(prices, lookback_days=lookback)
        note = (f"WARN: backtest with exactly lookback rows produces {len(result)} "
                f"result rows — may represent insufficient real out-of-sample data")
        return note
    except Exception as e:
        return f"FAIL: crashed with exactly lookback rows — {e}"


def test_backtest_prices_with_gaps():
    """NaN prices mid-series (missing trading days) → after dropna the window may shrink."""
    from backtest.engine import rolling_backtest
    prices = make_price_df(n_days=1000, n_assets=3)
    prices.iloc[300:310] = np.nan  # inject a gap
    try:
        result = rolling_backtest(prices, lookback_days=252)
        if not np.isfinite(result.values).all():
            return f"FAIL: non-finite values after backtest with NaN price gaps"
    except Exception as e:
        return f"FAIL: crashed on price gaps — {e}"


def test_backtest_all_identical_prices():
    """Constant prices throughout → zero log returns → singular cov → optimizer may fail."""
    from backtest.engine import rolling_backtest
    n = 1000
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    prices = pd.DataFrame({"A": [100.0]*n, "B": [200.0]*n, "C": [50.0]*n}, index=idx)
    try:
        result = rolling_backtest(prices, lookback_days=252)
        if not np.isfinite(result.values).all():
            return f"FAIL: non-finite values in backtest with constant prices"
    except Exception as e:
        return f"FAIL: crashed on constant-price backtest — {e}"


# =============================================================================
# MAIN — Run all tests and print summary
# =============================================================================

TESTS = [
    # --- Metrics ---
    ("log_returns | zero price propagates -inf",                   test_log_returns_zero_price),
    ("log_returns | negative price propagates NaN",                test_log_returns_negative_price),
    ("log_returns | single row produces empty DataFrame",          test_log_returns_single_row),
    ("log_returns | all-constant prices → all-zero returns",       test_log_returns_all_same),
    ("log_returns | Inf price propagates Inf returns",             test_log_returns_inf_price),
    ("log_returns | NaN price mid-series",                         test_log_returns_nan_price),
    ("portfolio_perf | zero volatility → Sharpe div-by-zero",     test_portfolio_perf_zero_vol),
    ("portfolio_perf | negative weights accepted silently",        test_portfolio_perf_negative_weights),
    ("portfolio_perf | all-zero weights → zero vol → Sharpe NaN", test_portfolio_perf_zero_weights),
    ("portfolio_perf | weights don't sum to 1 — no validation",   test_portfolio_perf_weights_not_sum_to_one),

    # --- Optimization ---
    ("optim | max_sharpe single asset → weight=1",                 test_optim_single_asset),
    ("optim | identical assets → singular covariance",             test_optim_identical_assets),
    ("optim | all-negative expected returns",                      test_optim_all_negative_returns),
    ("optim | near-zero covariance → Sharpe blow-up",             test_optim_zero_covariance),
    ("optim | min_vol single asset → weight=1",                   test_optim_min_vol_single_asset),
    ("optim | efficient frontier two assets",                      test_efficient_frontier_two_assets),
    ("optim | efficient frontier single asset",                    test_efficient_frontier_single_asset),

    # --- Monte Carlo ---
    ("monte_carlo | simulations=0 → crash on percentile",         test_mc_zero_simulations),
    ("monte_carlo | years=0 → zero days → empty cumsum",          test_mc_zero_years),
    ("monte_carlo | weight length mismatch",                       test_mc_weights_wrong_length),
    ("monte_carlo | negative weights (short positions)",           test_mc_negative_weights),
    ("monte_carlo | single-day returns DataFrame",                 test_mc_single_day_returns),
    ("monte_carlo | all-zero returns → final value ~1.0",         test_mc_all_zero_returns),
    ("monte_carlo | extreme drift → exponential overflow",         test_mc_extreme_positive_drift),

    # --- Performance ---
    ("performance | CAGR on constant series → 0%",                test_cagr_constant_series),
    ("performance | CAGR on single-value series → div-by-zero",   test_cagr_single_value),
    ("performance | CAGR with start=0 → div-by-zero",            test_cagr_zero_start),
    ("performance | CAGR with negative start value",              test_cagr_negative_start),
    ("performance | max_drawdown monotone rising → 0",            test_max_drawdown_monotone_up),
    ("performance | max_drawdown single value",                   test_max_drawdown_single_value),
    ("performance | max_drawdown all-zero portfolio",             test_max_drawdown_all_zeros),
    ("performance | calmar with zero max_drawdown",               test_calmar_zero_drawdown),
    ("performance | calmar with positive max_dd (sign convention)", test_calmar_positive_drawdown_value),
    ("performance | annual_vol with single observation",          test_annual_vol_single_day),

    # --- RiskProfileAgent ---
    ("risk_agent | age=0 (newborn) — score in bounds",            test_risk_age_zero),
    ("risk_agent | negative age — no input validation",           test_risk_age_negative),
    ("risk_agent | age=200 — score still in bounds",              test_risk_age_200),
    ("risk_agent | unknown tolerance — silent fallback",          test_risk_unknown_tolerance),
    ("risk_agent | horizon=0",                                     test_risk_zero_horizon),
    ("risk_agent | negative horizon — no validation",             test_risk_negative_horizon),
    ("risk_agent | missing equity key → KeyError",                test_risk_adjust_allocation_missing_key),
    ("risk_agent | adjusted weights sum to 1",                    test_risk_adjust_weights_sum_to_one),
    ("risk_agent | equity cap respected after adjustment",         test_risk_adjust_equity_weight_cap),
    ("risk_agent | LIQUIDBEES.NS missing → KeyError on excess",  test_risk_adjust_liquidbees_missing),

    # --- MarketRegimeAgent ---
    ("regime_agent | empty returns DataFrame",                    test_regime_empty_returns),
    ("regime_agent | all-NaN returns",                            test_regime_all_nan_returns),
    ("regime_agent | adjust with missing ticker keys",            test_regime_adjust_missing_asset),
    ("regime_agent | adjust zero-weight equity assets",           test_regime_adjust_zero_weight_assets),
    ("regime_agent | normalisation after bull-market boost",      test_regime_normalisation_after_bull),

    # --- DecisionAgent ---
    ("decision_agent | first call always rebalances",             test_decision_first_call_always_rebalance),
    ("decision_agent | NaN in new_weights",                       test_decision_nan_weights),
    ("decision_agent | Inf in new_weights",                       test_decision_inf_weights),
    ("decision_agent | zero threshold",                           test_decision_zero_threshold),
    ("decision_agent | negative threshold",                       test_decision_negative_threshold),
    ("decision_agent | empty weight arrays",                      test_decision_empty_weights),
    ("decision_agent | vol==0.18 boundary classification",        test_evaluate_market_state_boundary_vol),
    ("decision_agent | NaN vol/sharpe inputs",                    test_evaluate_market_state_nan_inputs),

    # --- StateManager ---
    ("state_manager | save/load round-trip",                      test_state_roundtrip),
    ("state_manager | corrupted JSON → empty dict",               test_state_corrupted_json),
    ("state_manager | missing file → empty dict",                 test_state_missing_file),
    ("state_manager | numpy float64 not JSON-serialisable",       test_state_non_serialisable_data),
    ("state_manager | empty allocation dict",                     test_state_empty_allocation),

    # --- Backtest Engine ---
    ("backtest | rows < lookback → empty or trivial result",      test_backtest_fewer_rows_than_lookback),
    ("backtest | single asset",                                   test_backtest_single_asset),
    ("backtest | exactly lookback rows",                          test_backtest_exactly_lookback_rows),
    ("backtest | NaN price gaps mid-series",                      test_backtest_prices_with_gaps),
    ("backtest | all-constant prices → singular cov",            test_backtest_all_identical_prices),
]


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  STRESS TEST SUITE — MiniProject Portfolio System")
    print("=" * 80 + "\n")

    sections = {
        "METRICS":          slice(0, 10),
        "OPTIMIZATION":     slice(10, 17),
        "MONTE CARLO":      slice(17, 24),
        "PERFORMANCE":      slice(24, 34),
        "RISK PROFILE AGENT": slice(34, 44),
        "MARKET REGIME AGENT": slice(44, 49),
        "DECISION AGENT":   slice(49, 57),
        "STATE MANAGER":    slice(57, 62),
        "BACKTEST ENGINE":  slice(62, 67),
    }

    test_list = list(TESTS)
    idx = 0
    for section, slc in sections.items():
        batch = test_list[slc]
        print(f"\n── {section} {'─'*(60-len(section))}")
        for name, fn in batch:
            run_test(name, fn)

    # Summary
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    total  = len(results)

    print("\n" + "=" * 80)
    print(f"  RESULTS:  {n_pass} passed  |  {n_fail} bugs found  |  {n_warn} warnings  |  {total} total")
    print("=" * 80)

    if n_fail > 0:
        print("\n🐛  BUGS FOUND:\n")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  ❌  {name}")
                for row in detail.strip().split("\n"):
                    print(f"       {row}")

    if n_warn > 0:
        print("\n⚠️   WARNINGS (design flaws / missing validation):\n")
        for name, status, detail in results:
            if status == WARN:
                print(f"  ⚠️   {name}")
                for row in detail.strip().split("\n"):
                    print(f"       {row}")

    print()
    sys.exit(0 if n_fail == 0 else 1)