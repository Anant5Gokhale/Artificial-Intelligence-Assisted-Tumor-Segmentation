'''Create Bounding-Box on an image. The Bounding-Box coordinates are returned in terminal'''

import cv2
i=0
# define a function to display the coordinates of

coord_list = list()
# of the points clicked on the image
def click_event(event, x, y, flags, params):

   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      coord_list.append((x,y))
      # put coordinates as text on the image
      cv2.putText(img, f'({x},{y})',(x,y),
      cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1)
      
      # draw point on the image
      cv2.circle(img, (x,y), 3, (0,255,255), -1)
   
   if (len(coord_list) )% 2== 0:
        for i in coord_list:
            x1, y1 = coord_list[0]
            x2, y2 = coord_list[1]
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            # white colour helps clear demarcation in gray-scale image 
            coord_list[:] = []
    
# read the input image
img = cv2.imread('/Users/anantgokhale/Desktop/Screenshot 2023-06-25 at 2.17.03 AM.png')

# create a window
cv2.namedWindow('Point Coordinates')

# bind the callback function to window
cv2.setMouseCallback('Point Coordinates', click_event)

# display the image
while True:
   cv2.imshow('Point Coordinates',img)
   cv2.resizeWindow("Resized_Window",img.shape[1] , img.shape[0])
   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break
cv2.destroyAllWindows()