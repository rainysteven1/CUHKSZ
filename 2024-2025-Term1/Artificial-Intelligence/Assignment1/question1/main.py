from typing import List

import numpy as np
import random

from graphviz import Digraph
from queue import Queue


class Node(object):

    def __init__(self, index: int, state: str) -> None:
        self.index = index
        self.state = state
        self.prior = None

    def set_prior(self, prior) -> None:
        self.prior = prior

    def reset(self):
        self.prior = None


class Graph(object):
    def __init__(
        self, states: List, edges: np.ndarray, start: int = 0, end: int = -1
    ) -> None:
        self.nodes = [Node(*item) for item in enumerate(states)]
        self.edges = edges
        self.start = start
        self.end = end

        self.num = len(states)
        self.flags = [False for _ in range(self.num)]
        self.visit_order = list()

    def reset(self) -> None:
        for node in self.nodes:
            node.reset()
        self.flags = [False for _ in range(self.num)]
        self.visit_order = list()

    def bfs(self) -> None:
        q = Queue()
        self.flags[self.start] = True
        self.visit_order.append(self.start)
        q.put(self.nodes[self.start])

        while not q.empty():
            temp = q.get()
            indices = list(
                filter(lambda x: self.edges[temp.index][x] == 1, range(self.num))
            )
            for idx in indices:
                if self.edges[temp.index][idx] == 1 and not self.flags[idx]:
                    self.flags[idx] = True
                    self.visit_order.append(idx)
                    self.nodes[idx].set_prior(temp)
                    q.put(self.nodes[idx])

    def dfs(self, randomed: bool = False) -> None:
        def _dfs(index: int) -> None:
            self.visit_order.append(index)
            self.flags[index] = True
            indices = list(filter(lambda x: self.edges[index][x] == 1, range(self.num)))
            if randomed:
                random.shuffle(indices)
            for idx in indices:
                if self.edges[index][idx] == 1 and not self.flags[idx]:
                    self.nodes[idx].set_prior(self.nodes[index])
                    _dfs(idx)

        _dfs(self.start)

    def _get_path(self) -> List:
        path = list()
        temp = self.nodes[-1]
        while temp:
            path.append(temp)
            temp = temp.prior
        return list(reversed(path))

    def draw(self, name: str) -> None:
        path = self._get_path()
        dot = Digraph(comment=name)
        print([self.nodes[idx].state for idx in self.visit_order])

        for idx in self.visit_order:
            node = self.nodes[idx]
            dot.node(
                node.state,
                node.state,
                color="red" if node in path else "black",
                shape="circle",
            )

        for node in self.nodes:
            if node.prior:
                dot.edge(
                    node.prior.state,
                    node.state,
                    color="red" if node in path else "black",
                )

        dot.render(name, format="png", view=True)


if __name__ == "__main__":
    random.seed(42)

    states = ["S", "a", "b", "c", "d", "e", "f", "h", "p", "q", "r", "G"]
    edges = np.zeros((len(states), len(states)), dtype=int)
    edges[0][4] = 1
    edges[0][5] = 1
    edges[0][8] = 1
    edges[2][1] = 1
    edges[3][1] = 1
    edges[4][2] = 1
    edges[4][5] = 1
    edges[5][3] = 1
    edges[5][7] = 1
    edges[5][-2] = 1
    edges[6][3] = 1
    edges[6][-1] = 1
    edges[7][8] = 1
    edges[7][9] = 1
    edges[8][9] = 1
    edges[-2][6] = 1

    g = Graph(states, edges)
    g.bfs()
    g.draw("BFS")
    g.reset()
    g.dfs()
    g.draw("DFS")
    g.reset()
    g.dfs(randomed=True)
    g.draw("DFS_random")
