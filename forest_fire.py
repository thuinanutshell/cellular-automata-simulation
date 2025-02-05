#!/usr/bin/env python
# coding: utf-8

# ### System Rule
# 1. A squared matrix (grid) - CA space
# 2. Moore neighborhood (9-cell) & periodic boundary
# 3. Trees are distributed with some given probability $p$ ($p=0$ means there are no trees and $p=1$ means trees are everywhere with no open space)
# 4. Set fire to one of the trees in this forest.
# 5. A tree will catch fire if there is *at least one tree burning* in its neighborhood.
# 6. The burning tree will be charred completely after one time step.

# ### Implementation

# In[59]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm


# In[23]:


grid = np.zeros((3,3))
grid.size


# In[35]:


random_indices = np.random.choice(a=range(grid.size), size=int(0.3*grid.size))
np.random.randint(0, 9)


# In[25]:


grid.flat[random_indices] = 1
grid


# In[70]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from IPython.display import HTML

class ForestFire:
    def __init__(self, size, density, radius):
        """Initialize the forest fire model.

        Args:
            size (int): The size of the CA space (grid will be size x size).
            density (float): Initial density of trees in the CA space (value between 0 and 1).
            radius (int): The radius of the neighborhood affecting fire spread.
        """
        self.size = size
        self.density = density
        self.radius = radius
        self.current_state = np.zeros((size, size))  # 0 = empty, 1 = tree, 2 = burning, 3 = burnt
        self.next_state = np.zeros((size, size))
        self.step_counter = 0

    def initialize(self):
        """Initialize the CA model with trees and a randomly ignited fire."""
        # Place trees based on density
        total_cells = self.size ** 2
        tree_indices = np.random.choice(total_cells, int(self.density * total_cells), replace=False)
        self.current_state.flat[tree_indices] = 1  # Trees are represented by 1

        # Select a random tree to ignite
        tree_coords = np.column_stack(np.where(self.current_state == 1))
        if tree_coords.size > 0:  # Ensure at least one tree exists
            fire_idx = np.random.randint(len(tree_coords))
            fire_x, fire_y = tree_coords[fire_idx]
            self.current_state[fire_x, fire_y] = 2  # Fire starts at this tree

    def draw(self):
        """Draw the current state of the cellular automaton."""
        plt.clf()
        cmap = matplotlib.colors.ListedColormap(['white', 'green', 'red', 'black'])  # 0=empty, 1=tree, 2=burning, 3=burnt
        plt.imshow(self.current_state, vmin=0, vmax=3, cmap=cmap)
        plt.title(f'Step {self.step_counter}')
        plt.axis('off')  # Remove axis for better visualization
        plt.show()

    def update(self):
        """Update the cellular automaton by applying fire spread rules."""
        for row in range(self.size):
            for col in range(self.size):
                state = self.current_state[row, col]

                if state == 2:  # Burning tree becomes burnt
                    state = 3
                elif state == 1:  # Tree might catch fire
                    for dr in range(-self.radius, self.radius + 1):
                        for dc in range(-self.radius, self.radius + 1):
                            if (dr != 0 or dc != 0):  # Ignore the tree itself
                                neighbor_x = (row + dr) % self.size
                                neighbor_y = (col + dc) % self.size
                                if self.current_state[neighbor_x, neighbor_y] == 2:  # If neighbor is burning
                                    state = 2
                                    break  # No need to check further

                self.next_state[row, col] = state

        # Swap states for the next iteration
        self.current_state, self.next_state = self.next_state, self.current_state
        self.step_counter += 1

def update(frame, sim, steps_per_frame, progress_bar):
    """Update function for animation."""
    for _ in range(steps_per_frame):
        sim.update()

    plt.cla()  # Clear the plot
    cmap = matplotlib.colors.ListedColormap(['white', 'green', 'red', 'black'])
    im = plt.imshow(sim.current_state, vmin=0, vmax=3, cmap=cmap)
    plt.title(f'Step {sim.step_counter}')
    plt.axis('off')
    progress_bar.update(1)

    return (im,)

def make_animation(sim, total_frames, steps_per_frame=1, interval=100):
    """Create an animation of the forest fire simulation."""
    sim.initialize()
    progress_bar = tqdm(total=total_frames)

    fig = plt.figure()

    animation = FuncAnimation(
        fig, update, fargs=(sim, steps_per_frame, progress_bar), 
        frames=total_frames, interval=interval, blit=False
    )

    output = HTML(animation.to_html5_video())
    plt.close(fig)  # Close figure to prevent duplicate rendering

    return output


# In[73]:


sim = ForestFire(size=100, density=0.5, radius=1)
make_animation(sim, total_frames=100, steps_per_frame=1)

