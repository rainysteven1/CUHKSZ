from typing import Dict, List, Tuple

import json
import numpy as np
import os
import random
import time

from graphviz import Digraph
from queue import Queue


class Node(object):

    def __init__(self, index: int, state: str) -> None:
        self.index = index
        self.state = state
        self.prior = None

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.state == other.state
        return False

    def set_prior(self, prior) -> None:
        self.prior = prior

    def reset(self):
        self.prior = None


class Graph(object):
    def __init__(
        self, states: List, edges: np.ndarray, start: str = "S", end: str = "G"
    ) -> None:
        self.nodes = [Node(*item) for item in enumerate(states)]
        self.edges = edges
        self.start = self.nodes[states.index(start)]
        self.end = self.nodes[states.index(end)]

        self.num = len(states)
        self.flags = [False] * self.num
        self.visit_order = list()

    def reset(self) -> None:
        for node in self.nodes:
            node.reset()
        self.flags = [False] * self.num
        self.visit_order = list()

    def bfs(self) -> None:
        q = Queue()
        q.put(self.start)

        while not q.empty():
            temp = q.get()
            self.visit_order.append(temp)

            if temp == self.end:
                break

            indices = list(
                filter(lambda x: self.edges[temp.index][x] == 1, range(self.num))
            )
            for idx in indices:
                if not self.flags[idx]:
                    self.nodes[idx].set_prior(temp)
                    q.put(self.nodes[idx])
                    self.flags[idx] = True

    def dfs(self, randomed: bool = False) -> None:
        def _dfs(node: Node) -> bool:
            self.visit_order.append(node)
            self.flags[node.index] = True

            if node == self.end:
                return True

            indices = list(
                filter(lambda x: self.edges[node.index][x] == 1, range(self.num))
            )
            if randomed:
                random.shuffle(indices)
            for idx in indices:
                if self.edges[node.index][idx] == 1 and not self.flags[idx]:
                    temp = self.nodes[idx]
                    temp.set_prior(self.nodes[node.index])
                    if _dfs(temp):
                        return True

            return False

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
        print([node.state for node in self.visit_order])

        for node in self.visit_order:
            dot.node(
                node.state,
                node.state,
                color="red" if node in path else "black",
                shape="circle",
            )

        for node in self.visit_order:
            if node.prior:
                dot.edge(
                    node.prior.state,
                    node.state,
                    color="red" if node in path else "black",
                )

        dot.render(os.path.join("result", name), format="png", view=True)


def read_data(filename: str) -> Tuple[np.ndarray, Dict]:
    with open(filename, "r") as f:
        data = json.load(f)
    edges = data["edges"]
    states = sorted({part for edge in edges for part in edge.split("-")})
    states.append(states.pop(0))

    length = len(states)
    matrix = np.zeros([length] * 2, dtype=int)
    for edge in edges:
        edge = edge.split("-")
        row = states.index(edge[0])
        col = states.index(edge[1])
        matrix[row, col] = 1
    return states, matrix


if __name__ == "__main__":
    random.seed(42)
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "data.json"))
    g = Graph(*read_data(filename))
    g.bfs()
    g.draw("BFS")
    g.reset()
    g.dfs()
    g.draw("DFS")
    g.reset()
    g.dfs(randomed=True)
    g.draw("DFS_random")
