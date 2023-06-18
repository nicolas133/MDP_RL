
import cv2
import numpy as np

def grid_map():
    map_data = cv2.imread('cropped_grid_maze.png', cv2.IMREAD_GRAYSCALE)# 0 represents black and 255 represents white.
    height, width = map_data.shape # Get height and Width in pixel
    grid_rows = 8
    grid_columns = 7
    cell_height = height // grid_rows
    cell_width = width // grid_columns


    Gridy_map = np.zeros((grid_rows, grid_columns), dtype=np.uint8)

    for y in range(grid_rows):
        print(f'row {y}')
        for x in range(grid_columns):
            print(f'column {x}')
            cell_start_x = x * cell_width
            cell_end_x = (x + 1) * cell_width
            cell_start_y = y * cell_height
            cell_end_y = (y + 1) * cell_height
            cell_space = map_data[cell_start_y:cell_end_y, cell_start_x:cell_end_x]#break pixels into cells


            if np.any(cell_space): #check if any black pixels .any returns true if 0 or None in array
                black_pixels_percentage = np.mean(cell_space == 0) #black cells are 0
                if black_pixels_percentage > 0.2:
                    cell_value = 1
                else:
                    cell_value = 0

                Gridy_map[y, x] = cell_value #update cell value with current value

                print('barrier here')
            else:
                print('all clear soldier')

    return Gridy_map
