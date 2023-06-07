#:D
import easyocr
import numpy as np
import rclpy
from rclpy.node import Node
import cv2
import os
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Float32MultiArray, Bool
import time
import imutils
from proyecto_interfaces.msg import Banner

class Analisis_Imagen(Node):

    def __init__(self):
        super().__init__('analisis_imagen')
        self.bridge=CvBridge()
        self.cap = cv2.VideoCapture('192.168.203.47:8080/video')
        print("Inicio del nodo que analiza la imagen recibida por la cámara")
        self.subscriber_move = self.create_subscription(Bool, 'tomar_foto' ,self.subscriber_callback, 5)
        self.publisher = self.create_publisher(Banner, 'vision/banner_group_12', 10)
        self.msg = Banner()
        self.reader = easyocr.Reader(["es"], gpu=True)
        self.flag = False
        self.banner = 0
        self.figure = "NA"
        self.word = "NA"
        self.color = "NA"
        self.color_r = (255,255,255)
        self.colors_list  = {'blue': [np.array([95, 255, 85]), np.array([120, 255, 255])],
          'red': [np.array([161, 165, 127]), np.array([178, 255, 255])],
          'yellow': [np.array([16, 0, 99]), np.array([39, 255, 255])],
          'green': [np.array([33, 19, 105]), np.array([77, 255, 255])]}

            
    
            
###########################################################3######################
    def subscriber_callback(self, msg):
        flag = msg.data()
        if flag:
            self.recibirIMG()
#####################################################################################
    def recibirIMG(self):
        if(self.cap.isOpened()):
            ret, frame = self.cap.read() 
            h, w, c = frame.shape
            print('width:  ', w)
            print('height: ', h)
            #cv_image = cv2.resize(frame, (int(h*0.5),int(w*0.5)))

        else: 
            self.cap.release()
            print("cap not opened")

        #image = frame[0:int (len(frame)*0.8),int (len(frame[0])*0.2): int (len(frame[0])*0.8)]
        self.detectar_colores()
        self.figure = self.detectar_figura(frame)
        self.detectar_letras(frame)
        #print (f"La figura es: {figura}")
        #ruta ="/home/sebastian/Uniandes202310/Robotica/proyecto_final/proyecto_final_ws/src/mi_robot_vision/mi_robot_vision/perspectiva_actual.png"
        self.msg.banner = self.banner
        self.msg.figure = self.figure
        self.msg.word = self.word
        self.msg.color = self.color
        self.publisher.publish(self.msg)

        cv2.imshow("Image window", frame)
        #cv2.imwrite (ruta,image)
        cv2.waitKey(10)
        tf = time.time()
        
#############################################################################################################
    def detectar_letras(self,image):
        result = self.reader.readtext(image, paragraph=True)
        print (f"len result: {len(result)}")
        for res in result:
            print("res:", res[1])
            self.word= res[1]
            pt0 = res[0][0]
            pt1 = res[0][1]
            pt2 = res[0][2]
            pt3 = res[0][3]
            cv2.rectangle(image, pt0, (pt1[0], pt1[1] - 23), (166, 56, 242), -1)
            cv2.putText(image, res[1], (pt0[0], pt0[1] -3), 2, 0.8, (255, 255, 255), 1)
            cv2.rectangle(image, pt0, pt2, (166, 56, 242), 2)
            cv2.circle(image, pt0, 2, (255, 0, 0), 2)
            cv2.circle(image, pt1, 2, (0, 255, 0), 2)
            cv2.circle(image, pt2, 2, (0, 0, 255), 2)
            cv2.circle(image, pt3, 2, (0, 255, 255), 2)
######################################################################################################################
    def detectar_figura(self,image): 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 20, 150)
        canny = cv2.dilate(canny, None, iterations=1)
        canny = cv2.erode(canny, None, iterations=1)
        cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 4
        figure="Circulo"
        n = 1
        for c in cnts:
            epsilon = 0.01*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            #print(len(approx))
            x,y,w,h = cv2.boundingRect(approx)
            if  n== len(cnts):
                if len(approx)==3:
                    figure="Triangulo"
                    cv2.putText(image,figure, (x,y-5),1,1.5,(0,255,0),2)
                elif len(approx)==4:
                    aspect_ratio = float(w)/h
                    print('aspect_ratio= ', aspect_ratio)
                    if aspect_ratio == 1:
                        figure="Cuadrado"
                        cv2.putText(image,figure, (x,y-5),1,1.5,(0,255,0),2)
                    else:
                        figure="Rectangulo"
                        cv2.putText(image,figure, (x,y-5),1,1.5,(0,255,0),2)
                elif len(approx)==5:
                    figure="Pentagono"
                    cv2.putText(image,figure, (x,y-5),1,1.5,(0,255,0),2)
                elif len(approx)==6:
                    figure="Hexagono"
                    cv2.putText(image,figure, (x,y-5),1,1.5,(0,255,0),2)
                elif len(approx)==7:
                    figure="Heptagono"
                    cv2.putText(image,figure, (x,y-5),1,1.5,(0,255,0),2)
                elif len(approx)==8:
                    figure="Octagono"
                    cv2.putText(image,figure, (x,y-5),1,1.5,(0,255,0),2)
                
                elif len(approx)>10:
                    figure="Circulo"
                    cv2.putText(image,figure, (x,y-5),1,1.5,(0,255,0),2)
            n +=1
        cv2.drawContours(image, [approx], 0, (0,255,0),2) 
        return figure
#################################################################################################################
    def detectar_colores(self):
        if self.cap.isOpened(): #main loop
            ret, frame = self.cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convertion to HSV
            for name, clr in self.colors_list.items(): # for each color in colors
                if self.find_color(hsv, clr):  # call find_color function above
                    c, cx, cy = self.find_color(hsv, clr)
                    cv2.drawContours(frame, [c], -1, self.color_r, 3) #draw contours
                    cv2.circle(frame, (cx, cy), 7, self.color_r, -1)  # draw circle
                    cv2.putText(frame, name, (cx,cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_r, 1)
                    self.color = name # put text
            #cv2.imshow("Frame: ", frame) # show image
        #cv2.waitkey(1)

    def find_color(frame, points):
        mask = cv2.inRange(frame, points[0], points[1])#create mask with boundaries 
        cnts = cv2.findContours(mask, cv2.RETR_TREE, 
                            cv2.CHAIN_APPROX_SIMPLE) # find contours from mask
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            area = cv2.contourArea(c) # find how big countour is
            if area > 5000  and area < 10000:      # only if countour is big enough, then
                M = cv2.moments(c)
                cx = int(M['m10'] / M['m00']) # calculate X position
                cy = int(M['m01'] / M['m00']) # calculate Y position
                return c, cx, cy
            
        '''
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
            if(area > 20000):
                print (area)
                self.color = "Red"
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y),
                                        (x + w, y + h),
                                        (0, 0, 255), 2)
                cv2.putText(image, "Red Colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255))	
        # Creating contour to track green color
        cnts, hierarchy = cv2.findContours(green_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        print(f"longitud contornos verde {len(cnts)}")
        for pic, contour in enumerate(cnts):
            area = cv2.contourArea(contour)
            if(area > 20000):
                print (area)
                self.color = "Green"
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y),
                                        (x + w, y + h),
                                        (0, 255, 0), 2)
                cv2.putText(image, "Green Colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0))
        # Creating contour to track blue color
        cnts, hierarchy = cv2.findContours(blue_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        print(f"longitud contornos azul {len(cnts)}")
        for pic, contour in enumerate(cnts):
            area = cv2.contourArea(contour)
            if(area > 20000):
                print (area)
                self.color = "Blue"
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y),
                                        (x + w, y + h),
                                        (255, 0, 0), 2)
                cv2.putText(image, "Blue Colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 0, 0))'''

                
def main(args=None):
    rclpy.init(args=args)
    analisis_imagen = Analisis_Imagen()
    rclpy.spin(analisis_imagen)
    analisis_imagen.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
