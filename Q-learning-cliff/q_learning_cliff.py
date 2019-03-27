import numpy as np


class QLearningGridWorld:
    def __init__(self, world, start_state, goal_state, terminals):
        self.world = world
        self.action_list = [0, 1, 2, 3]
        self.state_action_value = np.zeros([world.shape[0], world.shape[1], len(self.action_list)])

        self.start_state = start_state
        self.goal_state = goal_state
        self.state = None
        self.terminals = terminals
        self.terminals.append(goal_state)

    # choose action using epsilon greedy method
    def choose_action(self, state, epsilon):
        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(self.action_list)
        else:
            return np.argmax(self.state_action_value[state[0], state[1], :])

    def action2move(self, action):
        moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        return moves[action]

    def to_arrows(self, action):
        arrows = ['^', '>', 'v', '<']
        return arrows[action]

    def go(self, state, move):
        next_state = [None, None]
        next_state[0] = state[0] + move[0]
        next_state[1] = state[1] + move[1]
        if self.not_out_of_world(next_state):
            return next_state
        else:
            return state

    def not_out_of_world(self, state):
        if state[0] < 0 or state[1] < 0 or state[1] == self.world.shape[1] or state[0] == self.world.shape[0]:
            return False
        else:
            return True

    def init_state(self):
        self.state = self.start_state

    def get_reward_next_state(self, action):
        move = self.action2move(action)
        S_ = self.go(self.state, move)
        reward = self.world[S_[0], S_[1]]
        return S_, reward

    def get_max_Q(self, s_):
        return max(self.state_action_value[s_[0], s_[1], :])

    # core algorithm
    def q_learning_algorithm(self, epoch, epsilon, alpha, gamma):
        for i in range(0, epoch):
            self.init_state()
            while self.state not in self.terminals:
                action = self.choose_action(self.state, epsilon)
                s_, reward = self.get_reward_next_state(action)

                self.state_action_value[self.state[0], self.state[1], action] = \
                    (1 - alpha) * self.state_action_value[self.state[0], self.state[1], action] + \
                    alpha * (reward + gamma * self.get_max_Q(s_))

                self.state = s_

    def print_policy(self):
        policy_list = ''
        policy_table = np.zeros([self.world.shape[0], self.world.shape[1]])
        for x in range(0, policy_table.shape[0]):
            for y in range(0, policy_table.shape[1]):
                if [x, y] not in self.terminals:
                    policy_list += self.to_arrows(np.argmax(self.state_action_value[x, y, :]))
                    policy_list += '   '
                else:
                    policy_list += '.'
                    policy_list += '   '
            policy_list += '\n'
        print(policy_list)
