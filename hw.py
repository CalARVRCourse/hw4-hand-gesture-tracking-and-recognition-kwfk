'''@authors Frederick Kim, Joshua Yang
'''

import cv2
import argparse
import numpy as np
import pyautogui
import sys

window_name = 'Hand Tracker'
max_binary_value = 255


# Toggles for various gesture listeners

# --Simple gestures--

# Mouse movement
g_mouseMove = True

# Pinch to mouse drag
g_pinchMouseDrag = False

# Fist (0 fingers) to hit 'record' in Logic
g_fistRecord = True

# --Complex gestures--

# Number of fingers to play piano notes (https://musiclab.chromeexperiments.com/Shared-Piano/)
g_numFingers = False

# Area size of pinch ellipse to control zoom in and zoom out
g_pinchArea = False

# Pinch angle to rotate image
g_pinchAngle = True

# Hand slide up or down to flip through tracks in Logic
g_handSlide = False


def nothing(x):
    pass

def isIncreased(current, prev, threshold):
    return current > prev + threshold

def isDecreased(current, prev, threshold):
    return current < prev - threshold
    

cam = cv2.VideoCapture(2)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 turns OFF auto exp
cam.set(cv2.CAP_PROP_AUTO_WB, 0.25) # 0.25 turns OFF auto WB
cv2.namedWindow(window_name)

num_avg_frames = 4

prevCntAreaAvg = None
cntAreas = []
prevHandRingArea = None
handRingAreas = []
prevAngleAvg = None
angles = []
numContoursAvg = []
fingerCounts = []
prevTrackFingerAvg = None
trackFingers = []
prevHandXPos = None
handXPositions = []

isMouseDown = False

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    findContours = (cv2.getTrackbarPos('Contours', window_name) == 1)

        
    lower_HSV = np.array([0, 60, 0], dtype = "uint8")  
    upper_HSV = np.array([25, 255, 255], dtype = "uint8")  
    
    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)  
    
    
    lower_YCrCb = np.array((0, 138, 67), dtype = "uint8")  
    upper_YCrCb = np.array((255, 179, 133), dtype = "uint8")  
        
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  

    
    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb) 

    # bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
    # fgmask = bgModel.apply(frame)
    # skinMask = cv2.add(skinMask, fgmask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)  
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)  
    
    # blur the mask to help remove noise, then apply the  
    # mask to the frame  
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) 
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    # part 1: skin

    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    _, thresh_inv = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )  
    # part 3: no thresh binary invert
    _, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY+cv2.THRESH_OTSU )

    # part 2: thresh


    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh_inv,ltype=cv2.CV_16U)  
    markers = np.array(markers, dtype=np.uint8)  
    label_hue = np.uint8(179*markers/np.max(markers))  
    blank_ch = 255*np.ones_like(label_hue)  
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0

    # part 2: labeled_img

    if (ret>2):
        try:
            statsSortedByArea = stats[np.argsort(stats[:, 4])]
            roi = statsSortedByArea[-3][0:4]  
            x, y, w, h = roi  
            subImg = labeled_img[y:y+h, x:x+w]  
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);  
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  

            numContoursAvg.append(len(contours))
            if len(numContoursAvg) == num_avg_frames + 1:
                numContoursAvg = numContoursAvg[1:]

            maxCntLength = 0
            for i in range(0,len(contours)):  
                cntLength = len(contours[i])  
                if(cntLength>maxCntLength):  
                    cnt = contours[i]
                    maxCntLength = cntLength
            if(maxCntLength>=5):  
                ellipseParam = cv2.fitEllipse(cnt)
                (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)

                if g_pinchAngle == True:
                    angles.append(angle)
                    if len(angles) == num_avg_frames:
                        angleAvg = np.mean(angles)
                        if 'ringAngle' in sys.argv and prevAngleAvg is not None and isIncreased(angleAvg, prevAngleAvg, 30):
                            pyautogui.hotkey('command', 'r')
                            print('ring angle increase')
                        elif 'ringAngle' in sys.argv and prevAngleAvg is not None and isDecreased(angleAvg, prevAngleAvg, 30):
                            pyautogui.hotkey('command', 'l')
                            print('ring angle decrease')
                        prevAngleAvg = angleAvg
                        angles = []
                
                if g_pinchMouseDrag == True:
                    if 'mouseDown' in sys.argv and angle > 100:
                        print('mouse down')
                        # simple gesture: Register Mouse down/up
                        pyautogui.mouseDown()
                        isMouseDown = True
                    else:
                        # simple gesture: Register Mouse down/up
                        if isMouseDown == True:
                            print('mouse up')
                            pyautogui.mouseUp()
                            isMouseDown = False

                # print((x, y), (MA, ma))
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
                subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)  
            
            subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)

            if g_pinchArea == True:
                # ZOOM IN ZOOM OUT WITH HAND RING
                handRingArea = w * h
                handRingAreas.append(handRingArea)
                if len(handRingAreas) == num_avg_frames:
                    handRingAreaAvg = np.mean(handRingAreas)
                    if np.mean(numContoursAvg) > 2:
                        if 'ringArea' in sys.argv and prevHandRingArea is not None and isIncreased(handRingAreaAvg, prevHandRingArea, 500):
                            pyautogui.hotkey('command', '+')
                            print('ring area increase')
                        elif 'ringArea' in sys.argv and prevHandRingArea is not None and isDecreased(handRingAreaAvg, prevHandRingArea, 500):
                            pyautogui.hotkey('command', '-')
                            print('ring area decrease')
                    prevHandRingArea = handRingAreaAvg
                    handRingAreas = []


            _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
            contours=sorted(contours,key=cv2.contourArea,reverse=True)       
            if len(contours)>1:  
                largestContour = contours[0]  
                hull = cv2.convexHull(largestContour, returnPoints = False)

                M = cv2.moments(largestContour)
                cX = -10 + 3 * int(M["m10"] / M["m00"])
                cY = -250 + 3 * int(M["m01"] / M["m00"])
                if 'pointer' in sys.argv:
                    pyautogui.moveTo(cX, cY, duration=0.02, tween=pyautogui.easeInOutQuad)
                
                if g_handSlide == True:
                    # MOVE HAND LEFT OR RIGHT
                    xPos = cY
                    handXPositions.append(xPos)
    
                    if len(handXPositions) == num_avg_frames:
                        handXPosAvg = np.mean(handXPositions)
                        if 'handMove' in sys.argv and prevHandXPos is not None and isIncreased(handXPosAvg, prevHandXPos, 200):
                            pyautogui.hotkey('down')
                            print('down')
                        elif 'handMove' in sys.argv and prevHandXPos is not None and isDecreased(handXPosAvg, prevHandXPos, 200):
                            pyautogui.hotkey('up')
                            print('up')
                        prevHandXPos = handXPosAvg
                        handXPositions = []

                spacePressed = False

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

                            c_squared = (end[0] - start[0]) * 2 + (end[1] - start[1]) * 2  
                            a_squared = (far[0] - start[0]) * 2 + (far[1] - start[1]) * 2  
                            b_squared = (end[0] - far[0]) * 2 + (end[1] - far[1]) * 2  
                            angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    

                            if angle <= np.pi / 3:
                                fingerCount += 1
                                cv2.circle(frame, far, 4, [255, 0, 255], -1)
                        cv2.putText(frame, "Fingers: " + str(fingerCount), (20,20), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)

                        if g_numFingers == True:
                            fingerCounts.append(fingerCount)
                            if len(fingerCounts) == num_avg_frames + 4:
                                fingerCounts = fingerCounts[1:]
                            fingerCountAvg = np.mean(fingerCounts)
    
                            nearestFingerCount = round(fingerCountAvg)
                            if 'fingerTrack' in sys.argv and prevTrackFingerAvg is not None and isIncreased(nearestFingerCount, prevTrackFingerAvg, 0.9):
                                # pyautogui.hotkey('command', 'a')
                                if nearestFingerCount == 1:
                                    pyautogui.hotkey('a')
                                    print('a')
                                elif nearestFingerCount == 2:
                                    pyautogui.hotkey('s')
                                    print('s')
                                elif nearestFingerCount == 3:
                                    pyautogui.hotkey('d')
                                    print('d')
                                elif nearestFingerCount == 4:
                                    pyautogui.hotkey('f')
                                    print('f')
                            elif 'fingerTrack' in sys.argv and prevTrackFingerAvg is not None and isDecreased(nearestFingerCount, prevTrackFingerAvg, 0.9):
                                # pyautogui.hotkey('command', 'c')
                                if nearestFingerCount == 3:
                                    pyautogui.hotkey('g')
                                    print('g')
                                elif nearestFingerCount == 2:
                                    pyautogui.hotkey('h')
                                    print('h')
                                elif nearestFingerCount == 1:
                                    pyautogui.hotkey('j')
                                    print('j')
                                elif nearestFingerCount == 0:
                                    pyautogui.hotkey('k')
                                    print('k')
                                print('finger decrease')
                            prevTrackFingerAvg = nearestFingerCount


                        if g_handSlide == True:
                            if fingerCountAvg < 5.5 and fingerCountAvg > 3.5: 
                                cntAreas.append(cv2.contourArea(largestContour))
                            if len(cntAreas) == 2:
                                cntAreaAvg = np.mean(cntAreas)
                                if 'handArea' in sys.argv and prevCntAreaAvg is not None and isIncreased(cntAreaAvg, prevCntAreaAvg, 3000):
                                    pyautogui.hotkey('command', 'up')
                                    print('area increase')
                                elif 'handArea' in sys.argv and prevCntAreaAvg is not None and isDecreased(cntAreaAvg, prevCntAreaAvg, 3000):
                                    pyautogui.hotkey('command', 'down')
                                    print('area decrease')
                                prevCntAreaAvg = cntAreaAvg
                                cntAreas = []


                        if g_fistRecord == True:
                            if 'space' in sys.argv and fingerCountAvg < 0.5 and not spacePressed:
                                pyautogui.press('s')
                                # pyautogui.click()
                                print('space')
                                spacePressed = True
                            else:
                                spacePressed = False
                    
                    
            
            # --- part 1 ---
            # cv2.imshow(window_name, gray)
            
            # --- part 2 ---
            # cv2.imshow(window_name, labeled_img)
            # cv2.imshow("ROI "+str(2), subImg)  
            
            # --- part 3 ---
            cv2.imshow(window_name, frame)
            
        except:
            print("Error")
            
    k = cv2.waitKey(1) #k is the key pressed
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        cv2.destroyAllWindows()
        cam.release()
        break