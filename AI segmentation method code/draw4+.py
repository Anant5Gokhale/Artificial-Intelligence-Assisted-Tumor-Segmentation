'''This python script is used in all experiments1-11. This is the first file in the 2 stage pipeline:
<initialise-box-parameters>.py ----> <draw-masks-&-select-tumor>.py'''

'''IM is selected as the largest recognisable tumor region and its file path is stored in IM variable.
Using OpenCV a rectangular region is drawn around this largest tumor. This rectangle circumsribes an ellipse,
which is chosen as shape simmilar to tumor tissue'''

'''Ellipse has two use cases: 1) It encircles the roi leaving behind unnecesary regions
2) Mean intensity calulated is more representative of the tumor region  '''

'''Note: For OpenCV functions:
Width of image is abscissa or horizontal length. 
Height of image is ordinate or vertical length'''

'''Format of Output is:
Example:
Rectangle Coordinates (top-left) (bottom-right): ((117, 123), (190, 183))
Ellipse Area (pi*semi-major-axis*semi-minor-axis): 3392
Ellipse Info (Center: width,height) (Axes: major,minor): ((153, 153), (36, 30))
Result Tuple: (((117, 123), (190, 183)), 3392, ((153, 153), (36, 30)))'''

import cv2
import numpy as np


# Global variables
drawing = False
rectangle_started = False
rectangle_x, rectangle_y, rectangle_width, rectangle_height = -1, -1, -1, -1
line_start = (-1, -1)

center = (-1, -1)  # Initialize center, axes, and angle variables
axes = (-1, -1)
angle = -1

# Variables to store rectangle and ellipse information
rectangle_coordinates = ()
ellipse_area = -1
ellipse_info = ()
IM = '/Users/anantgokhale/local-git-repo-med-evaluation/dev-19aug/archive/BraTS2021_Training_Data/img00000-to-00005_t1ce/00005_100.png'
def draw_translucent_rectangle(event, x, y, flags, param):
    global drawing, rectangle_started, rectangle_x, rectangle_y, rectangle_width, rectangle_height, line_start, center, axes, angle, rectangle_coordinates, ellipse_area, ellipse_info, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rectangle_x, rectangle_y = x, y
        line_start = (x, y)
        rectangle_started = False

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img = cv2.imread(IM)  # Create a new black image for each frame

        # Create a transparent overlay image
        overlay = img.copy()
        cv2.rectangle(overlay, (rectangle_x, rectangle_y), (x, y), (0, 255, 0), -1)  # Filled rectangle

        # Calculate ellipse parameters
        rectangle_width, rectangle_height = x - rectangle_x, y - rectangle_y

        # Ensure width and height are positive
        if rectangle_width < 0:
            rectangle_x += rectangle_width
            rectangle_width = abs(rectangle_width)
        if rectangle_height < 0:
            rectangle_y += rectangle_height
            rectangle_height = abs(rectangle_height)

        if rectangle_width > rectangle_height:
            center = (rectangle_x + rectangle_width // 2, rectangle_y + rectangle_height // 2)
            axes = (rectangle_width // 2, rectangle_height // 2)
            angle = 0
        else:
            center = (rectangle_x + rectangle_width // 2, rectangle_y + rectangle_height // 2)
            axes = (rectangle_height // 2, rectangle_width // 2)
            angle = 90

        # Draw the ellipse on the overlay
        cv2.ellipse(overlay, center, axes, angle, 0, 360, (0, 0, 255), -1)  # Filled ellipse

        # Combine the overlay with reduced opacity (30%)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Calculate rectangle coordinates
        top_left = (rectangle_x, rectangle_y)
        bottom_right = (rectangle_x + rectangle_width, rectangle_y + rectangle_height)
        rectangle_coordinates = (top_left, bottom_right)

        # Calculate ellipse area
        ellipse_area = int(np.pi * axes[0] * axes[1])

        # Display the area near the mouse cursor
        text = f'Ellipse Area: {ellipse_area}'
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Store ellipse information
        ellipse_info = (center, axes)

        cv2.imshow('Rectangle and Ellipse', img)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False

        # Draw the final rectangle and ellipse on the image
        cv2.rectangle(img, (rectangle_x, rectangle_y), (rectangle_x + rectangle_width, rectangle_y + rectangle_height), (0, 255, 0), 2)
        cv2.ellipse(img, center, axes, angle, 0, 360, (0, 0, 255), 2)
        cv2.imshow('Rectangle and Ellipse', img)

# Create a black image
img = cv2.imread(IM)
cv2.imshow('Rectangle and Ellipse', img)

cv2.setMouseCallback('Rectangle and Ellipse', draw_translucent_rectangle)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Print rectangle coordinates, ellipse area, and ellipse info
print("Rectangle Coordinates:", rectangle_coordinates)
print("Ellipse Area:", ellipse_area)
print("Ellipse Info (Center, Axes):", ellipse_info)

# Store the information in a tuple
result_tuple = (rectangle_coordinates, ellipse_area, ellipse_info)
print("Result Tuple:", result_tuple)
