import json
import random
from pathlib import Path

ARMS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


class AdaptiveAgent:
    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        self.epsilon = epsilon
        self.q = {str(a): 0.0 for a in ARMS}
        self.n = {str(a): 0 for a in ARMS}
        self.rng = random.Random(seed)

    def best_arm(self) -> float:
        best_value = max(self.q.values())
        tied = [float(a) for a, v in self.q.items() if v == best_value]
        return min(tied, key=lambda a: abs(a - 0.5))

    def choose(self, explore: bool = False) -> float:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(ARMS)
        return self.best_arm()

    def update(self, arm: float, reward: float):
        key = str(arm)
        self.n[key] += 1
        self.q[key] += (reward - self.q[key]) / self.n[key]

    def save(self, path):
        Path(path).write_text(json.dumps({"epsilon": self.epsilon, "q": self.q, "n": self.n}, indent=2))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        agent = cls(epsilon=data["epsilon"])
        agent.q, agent.n = data["q"], data["n"]
        return agent
