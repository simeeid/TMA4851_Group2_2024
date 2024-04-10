import numpy as np
import cv2

class Game:
    def __init__(self, frame, screen_width, screen_height, cords):
        self.frame = frame
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.cords = cords  
        self.target_cords = np.array((cords[0][0]*self.screen_width, cords[0][1]*self.screen_height),dtype=int)
  

    