import requests


class ExplanationAgent:

    def generate_report(self, state, weights_dict, performance, horizon):
        prompt = f"""
    You are a disciplined institutional portfolio manager.

    STRICT RULES:
    - Use ONLY the numbers provided in Performance Metrics.
    - Do NOT modify or reinterpret numbers.
    - Do NOT invent any new numeric values.
    - Monte Carlo values are wealth multiples.
    - If Expected Final Value is 1.247, you must state 1.247 exactly.
    - Do NOT approximate or round differently.
    - No currency references.

    Portfolio Composition (Indian ETFs):
    - NIF100BEES: Large-cap equity ETF
    - MID150BEES: Mid-cap equity ETF
    - GOLDBEES: Gold ETF
    - LIQUIDBEES: Liquid ETF

    Market Regime: {state}
    Investment Horizon: {horizon} years

    Allocation:
    {weights_dict}

    Performance Metrics:
    {performance}

    Write a concise institutional investment memo covering:
    1. Allocation rationale given regime and risk profile
    2. Risk-adjusted performance (Sharpe, Calmar, Drawdown)
    3. Monte Carlo outlook (Expected value, VaR, CVaR, Probability of Loss)
    4. Balanced forward-looking risk commentary

    No currency references.
    No exaggeration.
    Be precise.
    """
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]