# Python code for Multiple Color Detection


import numpy as np
import cv2


# Capturing video through webcam
#webcam = cv2.VideoCapture(0)

# Start a while loop
#while(1):
	
	# Reading the video from the
	# webcam in image frames
	#_, image = webcam.read()

	# Convert the image in
	# BGR(RGB color space) to
	# HSV(hue-saturation-value)
	# color space

image = cv2.imread('prueba3.jpeg')
def detectar_colores(image):
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between image and mask determines
    # to detect only that particular color
    kernel = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(image, image,
                            mask = red_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(image, image,
                                mask = green_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(image, image,
                            mask = blue_mask)

    # Creating contour to track red color
    cnts, hierarchy = cv2.findContours(red_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    print(f"longitud contornos rojos {len(cnts)}")
    
    for pic, contour in enumerate(cnts):
        area = cv2.contourArea(contour)
        if(area > 1000):
            print (f"area {area}")
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y),
                                    (x + w, y + h),
                                    (0, 0, 255), 2)
            
            cv2.putText(image, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))
            if 1:
                break	

    # Creating contour to track green color
    cnts, hierarchy = cv2.findContours(green_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    print(f"longitud contornos verde {len(cnts)}")

    for pic, contour in enumerate(cnts):
        area = cv2.contourArea(contour)
        if(area > 1000):
            print (f"area {area}")
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y),
                                    (x + w, y + h),
                                    (0, 255, 0), 2)
            
            cv2.putText(image, "Green Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0))
            if 1:
                break

    # Creating contour to track blue color
    cnts, hierarchy = cv2.findContours(blue_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    print(f"longitud contornos azul {len(cnts)}")

    for pic, contour in enumerate(cnts):
        area = cv2.contourArea(contour)
        if(area > 1000):
            print (f"area {area}")
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y),
                                    (x + w, y + h),
                                    (255, 0, 0), 2)
            
            cv2.putText(image, "Blue f", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))
            if 1:
                break
            
    # Program Termination
    #cv2.imshow("Multiple Color Detection in Real-TIme", image)
    #cv2.waitKey(0)   
	#if cv2.waitKey(10) & 0xFF == ord('q'):
	#	cap.release()
	#	cv2.destroyAllWindows()
	#	break

detectar_colores(image)
cv2.imshow("Multiple Color Detection in Real-TIme", image)
cv2.waitKey(0)