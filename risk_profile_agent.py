class RiskProfileAgent:

    def __init__(self, age, horizon, tolerance):
        self.age = age
        self.horizon = horizon
        self.tolerance = tolerance.lower()

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

        tolerance_factor = tolerance_map.get(self.tolerance, 0.6)

        risk_score = (0.4 * age_factor +
                      0.3 * horizon_factor +
                      0.3 * tolerance_factor)

        return round(risk_score, 2)

    def adjust_allocation(self, weights_dict):

        risk_score = self.compute_risk_score()

        # Cap equity exposure
        max_equity = 0.8 * risk_score

        equity_assets = ["NIF100BEES.NS", "MID150BEES.NS"]

        equity_weight = sum(weights_dict[a] for a in equity_assets)

        if equity_weight > max_equity:

            scale = max_equity / equity_weight

            for a in equity_assets:
                weights_dict[a] *= scale

            # Move excess to Liquid
            excess = 1 - sum(weights_dict.values())
            weights_dict["LIQUIDBEES.NS"] += excess

        return weights_dict, risk_score