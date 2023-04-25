import cv2
import numpy as np
  
# Read image.
img = cv2.imread('.\Testbilder\VonOben.jpg', cv2.IMREAD_COLOR)

#img = cv2.VideoCapture(0)
#l_b=np.array([0,150,100])# lower hsv bound for red
#u_b=np.array([255,255,255])# upper hsv bound to red

l_b=np.array([0,150,100])# lower hsv bound for red
u_b=np.array([5,255,255])# upper hsv bound to red


# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", cv2.resize(gray,[400,200],4))
  
# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))
cv2.imshow("gray_blured", cv2.resize(gray_blurred,[400,200],4))

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(hsv,l_b,u_b)
cv2.imshow("mask",mask)

#binery_img = cv2.binary
  
# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
               param2 = 30, minRadius = 10, maxRadius = 100)
  
# Draw circles that are detected.
if detected_circles is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
  
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
  
        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
        cv2.putText(img,str(pt),[a,b],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        print(a,b)
cv2.imshow("Detected Circle", img)
        

cv2.waitKey(0)