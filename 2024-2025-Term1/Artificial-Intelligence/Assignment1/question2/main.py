from typing import Dict, List, Tuple

import json
import numpy as np
import os
import time

from queue import PriorityQueue


INF = 999


def run_exec(func):
    def wrapper(*args, **kwargs):
        if isinstance(args[0], Graph):
            start = time.time_ns()
            func(*args, **kwargs)
            end = time.time_ns()
            print(f"Time taken: {end - start} nanoseconds")
            print(args[0].get_optimal_path())
            args[0].reset()

    return wrapper


class Node(object):
    def __init__(self, index: int, state: str, hcost: int) -> None:
        self.index = index
        self.state = state
        self.hcost = hcost
        self.reset()

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.state == other.state
        return False

    def set_gcost(self, gcost: int) -> None:
        self.gcost = gcost

    def set_prior(self, prior) -> None:
        self.prior = prior

    def refresh_fcost(self) -> None:
        self.fcost = self.gcost + self.hcost

    def reset(self):
        self.set_gcost(INF)
        self.refresh_fcost()
        self.prior = None


class Graph(object):

    def __init__(self, edges: np.ndarray, hcosts: Dict, start: str, end: str) -> None:
        keys = list(hcosts.keys())
        self.nodes, self.edges = [
            Node(*item, hcost=hcosts[item[1]]) for item in enumerate(keys)
        ], edges
        self.start, self.end = (
            self.nodes[keys.index(start)],
            self.nodes[keys.index(end)],
        )

        self.num = len(self.nodes)
        self.flags = [False] * self.num

    def reset(self) -> None:
        for node in self.nodes:
            node.reset()
        self.flags = [False] * self.num

    @run_exec
    def greedy_best_first_search(self) -> None:
        q = PriorityQueue()
        q.put((self.start.hcost, self.start))

        while not q.empty():
            temp = q.get()[1]
            self.flags[temp.index] = True

            if temp == self.end:
                break

            for idx in range(self.num):
                if self.edges[temp.index][idx] != 0 and not self.flags[idx]:
                    node = self.nodes[idx]
                    node.set_prior(temp)
                    q.put((node.hcost, node))

    @run_exec
    def a_star_search(self) -> None:
        q = PriorityQueue()
        self.start.set_gcost(0)
        self.start.refresh_fcost()
        q.put((self.start.fcost, self.start))

        while not q.empty():
            temp = q.get()[1]
            self.flags[temp.index] = True

            if temp == self.end:
                break

            for idx in range(self.num):
                if self.edges[temp.index][idx] != 0 and not self.flags[idx]:
                    node = self.nodes[idx]
                    node.set_prior(temp)
                    node.set_gcost(temp.gcost + self.edges[temp.index][idx])
                    node.refresh_fcost()
                    q.put((node.fcost, node))

    def get_optimal_path(self) -> Tuple[List, int]:
        path = list()
        temp = self.end
        cost = 0
        while temp:
            path.append(temp.state)
            if temp.prior:
                cost += self.edges[temp.prior.index][temp.index]
            temp = temp.prior
        return list(reversed(path)), cost


def read_data(filename: str) -> Tuple[np.ndarray, Dict]:
    with open(filename, "r") as f:
        data = json.load(f)
    edges, hcosts = data.values()
    states = sorted({part for key in edges.keys() for part in key.split("-")})

    length = len(states)
    matrix = np.zeros([length] * 2, dtype=int)
    for item in edges.items():
        edge = item[0].split("-")
        row = states.index(edge[0])
        col = states.index(edge[1])
        matrix[row, col] = matrix[col, row] = item[1]

    hcosts = dict([item for item in hcosts.items() if item[0] in states])
    return matrix, hcosts


if __name__ == "__main__":
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "data.json"))
    g = Graph(*read_data(filename), "O", "B")
    g.greedy_best_first_search()
    g.a_star_search()
