from q_learning_cliff import *


HEIGHT = 4
WIDTH = 12

if __name__ == '__main__':
    cliff = np.zeros([HEIGHT, WIDTH])
    cliff[:, :] = -1
    cliff[3, 1:11] = -1000
    cliff[3, 11] = 1000

    terminals = []
    for y in range(1, 11):
        terminals.append([3, y])

    q_learning_grid_world = QLearningGridWorld(world=cliff,
                                               start_state=[3, 0],
                                               goal_state=[3, 11],
                                               terminals=terminals)

    q_learning_grid_world.q_learning_algorithm(epoch=10000, epsilon=0.2, alpha=0.5, gamma=1)
    q_learning_grid_world.print_policy()







