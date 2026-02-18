import json
import os
from datetime import datetime


class StateManager:

    def __init__(self, filepath="memory/portfolio_state.json"):
        self.filepath = filepath

    def load_state(self):
        if not os.path.exists(self.filepath):
            return {}

        with open(self.filepath, "r") as f:
            try:
                return json.load(f)
            except:
                return {}

    def save_state(self, allocation, regime, performance, risk_score):

        state = {
            "timestamp": datetime.now().isoformat(),
            "allocation": allocation,
            "regime": regime,
            "performance": performance,
            "risk_score": risk_score
        }

        with open(self.filepath, "w") as f:
            json.dump(state, f, indent=4)