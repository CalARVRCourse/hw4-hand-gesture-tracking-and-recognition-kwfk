from __future__ import print_function
import cv2
import argparse
import numpy as np

window_name = 'Hand Tracker'
isColor = False
max_binary_value = 255

def nothing(x):
    pass
    

cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 turns OFF auto exp
cam.set(cv2.CAP_PROP_AUTO_WB, 0.25) # 0.25 turns OFF auto WB
cv2.namedWindow(window_name)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    findContours = (cv2.getTrackbarPos('Contours', window_name) == 1)
    
    #convert to grayscale
    if isColor == False:
        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    lower_HSV = np.array([0, 80, 0], dtype = "uint8")  
    upper_HSV = np.array([25, 255, 255], dtype = "uint8")  
      
    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)  
      
      
    lower_YCrCb = np.array((0, 138, 67), dtype = "uint8")  
    upper_YCrCb = np.array((255, 179, 133), dtype = "uint8")  
          
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  

      
    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb) 
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)  
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)  
      
    # blur the mask to help remove noise, then apply the  
    # mask to the frame  
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) 
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    # part 1: skin

    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )  
    # part 3: no thresh binary invert
    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_OTSU )  

    # part 2: thresh
    output = thresh
    cv2.imshow(window_name, output)
            
    k = cv2.waitKey(1) #k is the key pressed
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        cv2.destroyAllWindows()
        cam.release()
        break

