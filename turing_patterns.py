import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm

class TuringPatterns:
  def __init__(self, 
               size: int, 
               density: float,
               short_radius: int,
               long_radius: int,
               short_weight: float,
               long_weight: float):
    self.size = size # the number of cells in one row 
    self.density = density # proportion of cells to activate
    self.current_state = np.zeros((size, size)) # a squared matrix
    self.next_state = np.zeros((size, size))

    self.Ra = short_radius # radius of the short-activation R_a
    self.Ri = long_radius # radius of the long-inhibation R_i

    self.wa = short_weight # weights of short activation
    self.wi = long_weight # weights of long inhibation

    self.step_counter = 0

  def initialize(self):
    # moore neighborhood
    num_cells = self.size ** 2
    # randomly choose which index to be active
    random_indices = np.random.choice(a=range(num_cells),
                                      size=int(round(self.density*num_cells, 1) + 1),
                                      replace=False) 
    
    self.current_state.flat[random_indices] = 1 # assigned the values of the indices to 1
    self.figure, self.axes = plt.subplots()
  
  def draw(self):
    '''
    Draw the current state of the cellular automaton.
    '''
    plot = self.axes.imshow(
        self.current_state, vmin=0, vmax=1, cmap='gray') # Change to 'gray'
    self.axes.set_title(f'State at step {self.step_counter}')
    return plot

  def update(self):
    for col in range(self.size):
      for row in range(self.size):
        state = self.current_state[row, col]
        na = ni = 0

        # Compute the sum of short-activation
        for col1 in range(-self.Ra, self.Ra+1): # start from the edge
          for row1 in range(-self.Ra, self.Ra+1): # start from the edge
            na += self.current_state[(row+row1)%self.size, (col+col1)%self.size]
        
        # Compute the sum of long-inhibition
        for col2 in range(-self.Ri, self.Ri+1):
          for row2 in range(-self.Ri, self.Ri+1):
            ni += self.current_state[(row+row2)%self.size, (col+col2)%self.size]
        
        diff = self.wa*na - self.wi*ni
        if diff > 0:
          state = 1
        else:
          state = 0
        self.next_state[row, col] = state
    
    self.current_state, self.next_state = self.next_state, self.current_state
    self.step_counter += 1
    
def update(frame_number, steps_per_frame, progress_bar):
  for _ in range(steps_per_frame):
    sim.update()
    progress_bar.update(1)
    return [sim.draw()]
    
def make_animation(sim, total_frames, steps_per_frame=1, interval=100):
    frame_number = 0
    sim.initialize()
    progress_bar = tqdm(total=total_frames)
    update(frame_number, steps_per_frame, progress_bar)
    animation = FuncAnimation(
        sim.figure, update, fargs=(steps_per_frame, progress_bar), init_func=lambda: [], frames=total_frames, interval=interval, blit=True) # Pass progress_bar using fargs
    output = HTML(animation.to_html5_video())
    sim.figure.clf()
    plt.close(sim.figure)
    return output

sim = TuringPatterns(size=50, 
                     density=0.5,
                     short_radius=1,
                     long_radius=5,
                     short_weight=1,
                     long_weight=0.1)
make_animation(sim, total_frames=20, steps_per_frame=1)