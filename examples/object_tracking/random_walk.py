import numpy as np
from matplotlib import pyplot as plt


def random_step(current_x, current_y, tendency_x=0.5):
    if np.random.binomial(1, tendency_x):
        return current_x + 1, current_y
    else:
        return current_x, current_y + 1


def fixed_length_random_walk(start_x=0, start_y=0, length=10, tendency_x=0.5):
    positions = np.zeros(shape=(length, length))
    positions[start_x, start_y] = 1
    current_x, current_y = start_x, start_y
    for repetition in range(length - 1):
        current_x, current_y = random_step(current_x, current_y, tendency_x=tendency_x)
        positions[current_x, current_y] = 1
    return positions


def random_walk(start_x=0, start_y=0, length=10, tendency_x=0.5):
    positions = np.zeros(shape=(length, length))

    current_x, current_y = start_x, start_y
    while current_x < length and current_y < length:
        positions[current_x, current_y] = 1
        current_x, current_y = random_step(current_x, current_y, tendency_x=tendency_x)
    return positions


def mcmc(start_x=0, start_y=0, length=10, tendency_x=0.5, repetitions=100):
    positions = np.zeros(shape=(length, length))
    for sample in range(repetitions):
        positions += random_walk(start_x, start_y, length, tendency_x)
    return positions / repetitions


def visualize_positions(position_array):
    plt.imshow(position_array, cmap="coolwarm", vmin=-position_array.max(), vmax=position_array.max())
    plt.show()


if __name__ == "__main__":
    positions = mcmc(0, 0, 10, 0.7, 1000)
    print(positions)
    visualize_positions(positions)
