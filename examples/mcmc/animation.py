import matplotlib.pyplot as plt
from matplotlib import  animation

def animate_array(frames):
  fig, ax = plt.subplots()
  im = ax.imshow(frames[0], cmap='hot', vmin=0)  # Choose your desired colormap
  fig.colorbar(im)

  def animate(i):
    im.set_data(frames[i])
    return im

  anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=500)  # Adjust interval for frame rate
  plt.show()


import numpy as np

animate_array(np.random.random((10,5,5)))