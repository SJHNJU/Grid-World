import numpy as np


class GridWorld:
    def __init__(self, world, terminals, walls):
        self.actionlist = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.states = []
        self.terminals = terminals
        self.walls = walls
        self.value_map = {}
        self.reward = {}
        self.policy = {}
        self.rows, self.cols = world.shape[0], world.shape[1]

        for terminal in self.terminals:
            self.policy.update({terminal: None})

        for wall in self.walls:
            self.policy.update({wall:None})

        for x in range(0, self.rows):
            for y in range(0, self.cols):
                if world[x][y] is not None:
                    self.states.append((x, y))
                    self.value_map.update({(x, y): world[x][y]})
                    self.reward.update({(x, y): world[x][y]})

    def turn_left(self, a):
        return self.actionlist[self.actionlist.index(a) - 1]

    def turn_right(self, a):
        return self.actionlist[(self.actionlist.index(a) + 1) % 4]

    def go(self, state, a):
        state1 = (state[0]+a[0], state[1]+a[1])
        if state1 in self.states:
            return state1
        else:
            return state

    def A(self, state):
        if state in self.terminals:
            return [None]
        else:
            return self.actionlist

    def S_(self, state, a):
        if a is None:
            return [(0, state)]
        else:
            # Fun thing here! Change here to see the recommend route
            return [(0.89, self.go(state, a)),
                    (0.1, self.go(state, self.turn_left(a))),
                    (0.01, self.go(state, self.turn_right(a)))]

    def find_best_action(self, e_a):
        table_e = []
        table_a = []
        for e, a in e_a:
            table_a.append(a)
            table_e.append(e)
        return table_a[table_e.index(max(table_e))]

    def print_value_map(self):
        value_map = np.zeros((self.rows, self.cols))
        value_map[:][:] = None
        for s in self.value_map:
            value_map[s[0]][s[1]] = self.value_map[s]

        print(value_map)

    def to_arrow(self, a):
        if a is not None:
            arrow = {(0, 1): '>', (-1, 0): '^', (0, -1): '<', (1, 0): 'v'}
            return arrow[a]
        else:
            return '.'

    def print_policy(self):
        policy_list = ''
        for x in range(0, self.rows):
            for y in range(0, self.cols):
                state = (x, y)
                policy_list += self.to_arrow(self.policy[state])
                policy_list += '   '
            policy_list += '\n'

        print(policy_list)



