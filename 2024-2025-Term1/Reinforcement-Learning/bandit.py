import numpy as np


class _Process(object):
    """多臂老虎机算法基本框架"""

    def __init__(self, bandit) -> None:
        self.bandit = bandit
        self.counts = np.zeros(bandit.K)
        self.actions = list()
        self.regret = 0
        self.regrets = list()

    def update_regret(self, k: int) -> None:
        """计算累积懊悔并保存

        Args:
            k (int): 本次动作选择的拉杆的编号
        """
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self) -> int:
        """返回当前动作选择哪一根拉杆,由每个具体的策略实现"""
        raise NotImplementedError

    def run(self, num_steps: int) -> None:
        """运行

        Args:
            num_steps (int): 总运行次数
        """
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(_Process):
    """epsilon贪婪算法"""

    def __init__(self, bandit, epsilon: float = 0.01, init_prob: float = 1.0) -> None:
        super(_Process, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self) -> int:
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.bandit.probs)
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
