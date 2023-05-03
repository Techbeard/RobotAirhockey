import numpy as np
import cv2
import json
def resize(img):
        # arg1- input image, arg- output_width, output_heightq
        return cv2.resize(img, (512, 512))

def doTracking(frame):

# with open('config.json','r') as file:
#     config = json.load(file)

# print(config["camera"])




# cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

# while True:
    

    # rat,frame = cap.read()
    #cv2.imshow("Camera", resize(frame))
 

   # l_b=np.array([0,150,100])# lower hsv bound for red
    #u_b=np.array([5,255,255])# upper hsv bound to red

    #hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #mask=cv2.inRange(hsv,l_b,u_b)
    #cv2.imshow("mask",mask)

    # Convert to grayscale.
    color = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2GRAY)
    
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray, 
                       cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                   param2 = 30, minRadius = 60, maxRadius = 80)
    
    # Draw circles that are detected.
    if detected_circles is not None:
            
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
    
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            #outpt = ''
            #outpt = pt
            # Draw the circumference of the circle.
            cv2.circle(color, (a, b), r, (0, 255, 0), 2)
            cv2.putText(color,str(pt),[a,b],cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255))
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(color, (a, b), 1, (0, 0, 255), 3)
            print(a,b)
            #cv2.imshow("Detected Circle", resize(frame))

    cv2.imshow("Detected Circle", color)         

    # key=cv2.waitKey(1)
    # if key==ord('q'):
    #     break
# cv2.waitKey(0)
# cv2.destroyAllWindows()