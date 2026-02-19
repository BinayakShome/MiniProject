class RiskProfileAgent:

    VALID_TOLERANCES = {"low", "medium", "high"}

    def __init__(self, age, horizon, tolerance):
        if not isinstance(age, (int, float)) or age < 0:
            raise ValueError(f"age must be a non-negative number, got {age!r}")
        if not isinstance(horizon, (int, float)) or horizon < 0:
            raise ValueError(f"horizon must be a non-negative number, got {horizon!r}")
        tolerance = tolerance.lower()
        if tolerance not in self.VALID_TOLERANCES:
            raise ValueError(
                f"tolerance must be one of {self.VALID_TOLERANCES}, got {tolerance!r}"
            )
        self.age = age
        self.horizon = horizon
        self.tolerance = tolerance

    def compute_risk_score(self):

        # Age factor (younger = more risk)
        age_factor = max(0, min(1, (60 - self.age) / 40))

        # Horizon factor (longer = more risk)
        horizon_factor = max(0, min(1, self.horizon / 30))

        # Tolerance factor
        tolerance_map = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.9
        }

        tolerance_factor = tolerance_map[self.tolerance]

        risk_score = (0.4 * age_factor +
                      0.3 * horizon_factor +
                      0.3 * tolerance_factor)

        return round(risk_score, 2)

    def adjust_allocation(self, weights_dict):

        risk_score = self.compute_risk_score()

        # Cap equity exposure
        max_equity = 0.8 * risk_score

        equity_assets = ["NIF100BEES.NS", "MID150BEES.NS"]

        # Only consider equity assets that are actually present in the portfolio
        present_equity = [a for a in equity_assets if a in weights_dict]
        equity_weight = sum(weights_dict[a] for a in present_equity)

        if equity_weight > max_equity and equity_weight > 0:

            scale = max_equity / equity_weight

            for a in present_equity:
                weights_dict[a] *= scale

            excess = 1 - sum(weights_dict.values())

            if "LIQUIDBEES.NS" in weights_dict:
                weights_dict["LIQUIDBEES.NS"] += excess
            elif excess != 0:
                # Distribute excess proportionally among all non-equity assets
                non_equity = [k for k in weights_dict if k not in equity_assets]
                if non_equity:
                    per_asset = excess / len(non_equity)
                    for k in non_equity:
                        weights_dict[k] += per_asset
                else:
                    # Edge case: portfolio is 100% equity; add LIQUIDBEES.NS
                    weights_dict["LIQUIDBEES.NS"] = excess

        return weights_dict, risk_score