
from GridWorld import *


def value_iteration(world):
    old_value_map = world.value_map.copy()
    for s in world.states:
        # store expectation corresponding to the action
        e_a = []
        for a in world.A(s):
            e = 0
            for p, s_ in world.S_(s, a):
                e += p * old_value_map[s_]
            e_a.append(e)
        # update the value here!!!
        world.value_map[s] = world.reward[s] + max(e_a)


def generate_policy_from_value_map(world):
    for s in world.states:
        if s not in world.terminals:
            e_a = []
            for a in world.A(s):
                e = 0
                for p, s_ in world.S_(s, a):
                    e += p * world.value_map[s_]
                e_a.append((e, a))

            best_a = world.find_best_action(e_a)

            world.policy.update({s: best_a})


if __name__ == '__main__':
    grid_world = GridWorld(world=np.array([[-0.02, -0.02, -0.02, 1],
                                      [-0.02, None , -0.02, -1],
                                      [-0.02, -0.02, -0.02, -0.02]]),
                           terminals=[(0, 3), (1, 3)],
                           walls=[(1, 1)])

    for i in range(0, 40):
        value_iteration(grid_world)

    grid_world.print_value_map()

    generate_policy_from_value_map(grid_world)

    grid_world.print_policy()
