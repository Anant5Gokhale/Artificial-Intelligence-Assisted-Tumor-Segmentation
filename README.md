# openCV_healthcare

#VIDEO TUTORIAL:OPEN CV 
#https://www.youtube.com/watch?v=oXlwWbU8l2o

import cv2 as cv

img = cv.imread('') # Takes path to an image and returns a matrix of image

cv.imshow('') # displays image as window. Takes two inputs: name of image 
// and the matrix of image

cv.waitKey(0) # Waits for an input from keyboeard. Example: when input is 0
 it waits for infinite time

// In imshow large images tend to go offscreen. Now to read videos

capture = cv.VideoCapture('/path') # Input is path. For wec camera input is 0. While other
// succesive integers will choose other cameras connected to system.

while True:
    isTrue, frame  = capture_read() #read video frame by frame
    cv.imshow('Video' , frame)
    #to stop the video from playing indefinately...
    if cv.wait(20) & 0xFF == ord('d')
        break

capture.release()
cv.destroyAllWindows()    
// We might get error as video/photo runs out of frame.

#PART3: Resizing and Rescaling of images

def rescaleFrame(frame, scale =0.75):
    #for video, image, live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimmensions = (width, height)

    return cv.resize(frame, dimmensions, interpolation=cv.INTER_AREA)

capture = cv.VideoCapture('/path')

while True:
    isTrue, frame  = capture_read() 
    frame_resized = rescaleFrame(frame)
    cv.imshow('Video Resized', frame_resized)
    cv.imshow('Video' , frame)
    if cv.wait(20) & 0xFF == ord('d')
        break

capture.release()
cv.destroyAllWindows()

// We can also resize images the same way
img = cv.imread('')
image_resized = rescaleFrame(img) 
cv.imshow('Image', image_resized)

def changeRes(width, height):
    # Live video only(like webcam)
    capture.set(3,width)
    capture.set(4,height)

#PART4: DRAWING SHAPES AND PUTTING TEXT
import cv2 as cv
import numpy as np
img = cv.imread('path')
cv.imshow('Name of image', img)
# To create a blank image. So we can draw on it.
blank = np.zeros((500,500,3), dtype = 'uint8') # height, width and number of colur channels
cv.imsshow('Blank', blank)
#Give color to complete image
blank[:] = 0,255,0 #referce all points and set colour to green
cv.imshow('Green', blank)
#Create blank red square in image
blank[200:300, 300:400] = 0,0,255
cv.imshow('Red square in blank', blank)

#Draw a rectangle
cv.rectangle(blank,(0,0), (250,250),(250,0), thickness =2)# boundary thickness =2. no fill
cv.rectangle(blank,(0,0), (250,250),(250,0), thickness =cv.FILLED) #filled rectangle
cv.rectangle(blank,(0,0), (250,250),(250,0), thickness =-1) #filled rectangle
cv.rectangle(blank,(0,0), blank.shape[1]//2, blank.shape[0]//2, (0,255,0), thickess=-1)# filled square
#1/2height and 1/2width. Note integer division is used
cv.imshow('Rectangle', blank)

#Circle
cv.circle(blank, (250,250), 40, (0,255,0), thickness = 2)
#centre at 250,250, radius 40, colur red and thickness 2
cv.circle(blank, blank.shape[1]//2, blank.shape[0]//2, 40, (0,255,0), thickness = 2)
#same circle
cv.imshow('Circle', blank)

#Line
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thicknes=-1)
#white line
cv.imshow('line', blank)

#Text on image
cv.putText(blank, 'Hello', (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), thickness = 2)
#if textgoes out then change position

#PART5 : ESSENTIAL FUNCTIONS


import cv2 as cv
img = cv.imread('Photos/cat.jpg')
cv.imshow('Cat', img)
#converting an image to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('grey', gray )

#Blur. ie remove noise from image.
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT) #kernel has to be an odd number.
#larger number gives more blure
cv.imshow('blur', blur)

#edge cascade=
canny = cv.Canny(img, 125,175)
cv.imshow('Canny Edges', canny)
#edges can be reduced passing blur picture
cv.imshow('Canny Edges', blur)

# Dilating the image
dilated = cv.dilate(canny, (3,3), iterations=1)
cv.imshow('dilated', dilated)

#Eroding
eroded = cv.erode(dilated, (3,3), iterations=1)
cv.imshow('eroded', eroded)

#Resize
resize = cv.resize(img, (500,500))
cv.imshow('resize', resize)
resize = cv.resize(img, (500,500), interpolation= cv.INTER_CUBIC)
resize = cv.resize(img, (500,500), interpolation= cv.INTER_LINEAR)
resize = cv.resize(img, (500,500), interpolation= cv.INTER_AREA)

#Crop
cropped = img[50::200, 200,400]
cv.imshow('cropped', cropped)

# PART6: IMAGE TRANSLATION
 
