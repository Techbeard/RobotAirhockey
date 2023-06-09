import numpy as np
import cv2


def resize(img):
        # arg1- input image, arg- output_width, output_height
        return cv2.resize(img, (512, 512))


cap = cv2.VideoCapture("TestBilder\Testviedeo.mp4")

while True:
    

    rat,frame = cap.read()
    cv2.imshow("Camera", resize(frame))
    #img = cv2.VideoCapture(0)
   # l_b=np.array([0,150,100])# lower hsv bound for red
    #u_b=np.array([5,255,255])# upper hsv bound to red

   # l_b=np.array([200,0,20])# lower hsv bound for blue
    #u_b=np.array([255,150,30])# upper hsv bound to blueq

    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,l_b,u_b)
    cv2.imshow("mask",mask)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur using 3 * 3 kernel.
    #gray_blurred = cv2.blur(gray, (3, 3))
    
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray, 
                       cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                   param2 = 30, minRadius = 19, maxRadius = 23)
    
    # Draw circles that are detected.
    if detected_circles is not None:
            
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
    
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            #outpt = ''
            #outpt = pt
            # Draw the circumference of the circle.
            cv2.circle(frame, (a, b), r, (0, 255, 0), 2)
            cv2.putText(frame,str(pt),[a,b],cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255))
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)
            print(a,b)
            #cv2.imshow("Detected Circle", resize(frame))

    cv2.imshow("Detected Circle", resize(frame))         

    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()