import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation



# Sample data sequences for animations (replace with your actual data)
frames1 = []
for i in range(10):
  data1 = np.random.rand(5, 5)  # Sample data for animation 1
  frames1.append(data1)

frames2 = []
for i in range(20):
  data2 = np.sin(np.linspace(0, 10*i, 100))  # Sample data for animation 2
  frames2.append(data2)


def animate1(i):
  # Update data for animation 1
  im1.set_data(frames1[i])
  return im1

def animate2(i):
  # Update data for animation 2
  line2.set_ydata(frames2[i])
  return line2


# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2)  # 1 row, 2 columns

# Create plots for each animation
im1 = ax1.imshow(frames1[0], cmap='hot')  # Animation 1 (image)
line2, = ax2.plot(frames2[0])  # Animation 2 (line plot)
ax2.set_ylim(-1.2, 1.2)  # Set limits for line plot

# Create animations
anim1 = animation.FuncAnimation(fig, animate1, frames=len(frames1), interval=50, ax=ax1)
anim2 = animation.FuncAnimation(fig, animate2, frames=len(frames2), interval=100, ax=ax2)

# Adjust layout
plt.tight_layout()

plt.show()