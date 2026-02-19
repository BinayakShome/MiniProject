import requests


class ExplanationAgent:

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "phi3:mini"

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
        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=None
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "[No response field in Ollama reply]")

        except requests.exceptions.ConnectionError:
            return (
                f"[Memo unavailable — Ollama is not running]\n\n"
                f"To enable AI memos, install Ollama (https://ollama.com) and run:\n"
                f"  ollama serve\n"
                f"  ollama pull {self.MODEL}\n\n"
                f"--- Summary from computed metrics ---\n"
                f"Regime      : {state}\n"
                f"Horizon     : {horizon} years\n"
                f"Allocation  : {weights_dict}\n"
                f"Performance : {performance}"
            )
        except requests.exceptions.Timeout:
            return "[Memo unavailable — Ollama request timed out after 60 s]"
        except requests.exceptions.HTTPError as e:
            return f"[Memo unavailable — Ollama returned HTTP error: {e}]"
        except (KeyError, ValueError) as e:
            return f"[Memo unavailable — unexpected Ollama response format: {e}]"
