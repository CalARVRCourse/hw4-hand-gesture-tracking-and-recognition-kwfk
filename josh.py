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


    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh,ltype=cv2.CV_16U)  
    markers = np.array(markers, dtype=np.uint8)  
    label_hue = np.uint8(179*markers/np.max(markers))  
    blank_ch = 255*np.ones_like(label_hue)  
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0

    # part 2: labeled_img

    if (ret>2):  
        try:
            # statsSortedByArea = stats[np.argsort(stats[:, 4])]
            # roi = statsSortedByArea[-3][0:4]  
            # x, y, w, h = roi  
            # subImg = labeled_img[y:y+h, x:x+w]  
            # subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);  
            # _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            # maxCntLength = 0  
            # for i in range(0,len(contours)):  
            #     cntLength = len(contours[i])  
            #     if(cntLength>maxCntLength):  
            #         cnt = contours[i]  
            #         maxCntLength = cntLength  
            # if(maxCntLength>=5):  
            #     ellipseParam = cv2.fitEllipse(cnt)
            #     (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
            #     print((x, y), (MA, ma))
            #     subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
            #     subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)  
              
            # subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)

            _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
            contours=sorted(contours,key=cv2.contourArea,reverse=True)       
            if len(contours)>1:  
                largestContour = contours[0]  
                hull = cv2.convexHull(largestContour, returnPoints = False)
                for cnt in contours[:1]:  
                    defects = cv2.convexityDefects(cnt,hull)  
                    if(not isinstance(defects,type(None))):
                        fingerCount = 0
                        for i in range(defects.shape[0]):
                            s,e,f,d = defects[i,0]  
                            start = tuple(cnt[s][0])  
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            cv2.line(frame,start,end,[0,255,0],2)  
                            cv2.circle(frame,far,5,[0,0,255],-1)

                            c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
                            a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
                            b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
                            angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    

                            if angle <= np.pi / 3:  
                                fingerCount += 1  
                                cv2.circle(frame, far, 4, [255, 0, 255], -1)


            cv2.imshow(window_name, frame)
            # part 2
            # output = subImg
            # cv2.imshow("ROI "+str(2), output)  
        except:
            print('hi')
            output = thresh
            cv2.imshow(window_name, output)
            
    k = cv2.waitKey(1) #k is the key pressed
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        cv2.destroyAllWindows()
        cam.release()
        break
