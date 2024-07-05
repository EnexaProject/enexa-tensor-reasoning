import matplotlib.pyplot as plt
import numpy as np


def visualize_sudoku(sudoku, number=3):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Create a table
    table = plt.table(cellText=sudoku,
                      cellLoc='center',
                      loc='center',
                      cellColours=[['white'] * number ** 2 for _ in range(number ** 2)],
                      colWidths=[0.1] * number ** 2)

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    # Adjust the table
    if number == 3:
        table.scale(1.01, 2.8)
    elif number == 2:
        table.scale(2.3, 6.3)
    # Draw grid lines
    for i in range(number ** 2 + 1):
        lw = 2 if i % number == 0 else 0.5
        ax.plot([0, number ** 2], [i, i], color='black', lw=lw)
        ax.plot([i, i], [0, number ** 2], color='black', lw=lw)

    plt.show()