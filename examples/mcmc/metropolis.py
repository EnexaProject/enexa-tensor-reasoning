import numpy as np

import matplotlib.pyplot as plt
from matplotlib import  animation

def get_neighbors(x_pos, y_pos, size):
    if x_pos > 0 and x_pos < size[0] - 1:
        if y_pos > 0 and y_pos < size[1] - 1:
            return 9
        else:
            return 6
    else:
        if y_pos > 0 and y_pos < size[1] - 1:
            return 6
        else:
            return 4

def random_neighbor(x_pos, y_pos, size):
    if x_pos == size[0] - 1:
        x_new = np.random.choice([x_pos, x_pos - 1])
    elif x_pos == 0:
        x_new = np.random.choice([x_pos, x_pos + 1])
    else:
        x_new = np.random.choice([x_pos - 1, x_pos, x_pos + 1])

    if y_pos == size[1] - 1:
        y_new = np.random.choice([y_pos, y_pos - 1])
    elif x_pos == 0:
        y_new = np.random.choice([y_pos, y_pos + 1])
    else:
        y_new = np.random.choice([y_pos - 1, y_pos, y_pos + 1])

    if x_new == -1 or y_new == -1:
        return x_pos, y_pos
    return x_new, y_new

def metropolis_step(x_pos, y_pos, prob):
    x_new, y_new = random_neighbor(x_pos, y_pos, prob.shape)

    acceptance_prob = min(1, (get_neighbors(x_pos, y_pos, prob.shape) / get_neighbors(x_new, y_new, prob.shape)) * (
                prob[x_new, y_new] / prob[x_pos, y_pos]))

    return (x_new, y_new) if np.random.rand() < acceptance_prob else (x_pos, y_pos)

def metropolis_markov_chain(x_start, y_start, prob, chainlength):
    positions = np.empty((chainlength, 2))
    positions[0,:] = x_start, y_start

    x_pos, y_pos = x_start, y_start
    for step in range(1, chainlength):
        x_pos, y_pos = metropolis_step(x_pos, y_pos, prob)
        positions[step, :] = x_pos, y_pos

    return positions

def positions_to_3D_array(positions, x_size, y_size):
    out = np.zeros(shape=(positions.shape[0], x_size, y_size))
    for i in range(positions.shape[0]):
        out[i, int(positions[i,0]), int(positions[i, 1])] = 1
    return out

def average_3D_array(positions):
    out = np.empty(positions.shape)
    for i in range(positions.shape[0]):
        out[i] = positions[:i+1].mean(axis=0)
    return out

def animate_array(frames1, frames2, image, interval=100):
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,5))
  ax4.axis("off")
  maxval = image.max()

  im1 = ax1.imshow(frames1[0], cmap='hot', vmin=0, vmax=maxval)  # Choose your desired colormap
  ax1.set_title("Walk")
  fig.colorbar(im1, label="Probability density")

  ax2.set_title("Averaged Walk")
  im2 = ax2.imshow(frames2[0], cmap='hot', vmin=0, vmax=maxval)  # Choose your desired colormap

  ax3.set_title("Distribution")
  ax3.imshow(image, cmap="hot", vmin=0, vmax=maxval)
  def animate1(i):
    im1.set_data(frames1[i])
    return im1

  def animate2(i):
    im2.set_data(frames2[i])
    return im2

  ani = animation.FuncAnimation(fig, animate1, frames=len(frames1), interval=interval)  # Adjust interval for frame rate
  ani2 = animation.FuncAnimation(fig, animate2, frames=len(frames2), interval=interval)  # Adjust interval for frame rate

  plt.show()

def gaussian(x, y, mu_x, mu_y, sigma):
    return np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))

def gaussian_grid(x_size, y_size):
    values = np.empty((x_size,y_size))
    for x_pos in range(x_size):
        for y_pos in range(y_size):
            values[x_pos, y_pos] = gaussian(x_pos, y_pos, x_size/2, y_size/2, 1)
    return values/ values.sum()

if __name__ == "__main__":
    prob = gaussian_grid(5,5)

    walk = positions_to_3D_array(metropolis_markov_chain(0,0,prob,10),*prob.shape)

    animate_array(walk, average_3D_array(walk), prob, interval=10)