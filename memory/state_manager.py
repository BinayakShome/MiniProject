import json
import os
import numpy as np
from datetime import datetime


class _NumpyEncoder(json.JSONEncoder):
    """Safely serialise numpy scalars and arrays to plain Python types."""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class StateManager:

    def __init__(self, filepath="memory/portfolio_state.json"):
        self.filepath = filepath

    def load_state(self):
        if not os.path.exists(self.filepath):
            return {}

        with open(self.filepath, "r") as f:
            try:
                return json.load(f)
            except Exception:
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
            json.dump(state, f, indent=4, cls=_NumpyEncoder)