import easyocr
import numpy as np
import rclpy
from rclpy.node import Node
import cv2
import os
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

class Analisis_Imagen(Node):

    def __init__(self):
        super().__init__('analisis_imagen')
        self.bridge=CvBridge()
        self.cap = cv2.VideoCapture('192.168.210.65:8080/video')
        print("Inicio del nodo que analiza la imagen recibida por la cámara")
        self.reader = easyocr.Reader(["es"], gpu=False)
        while(rclpy.Ok()):
            recibirIMG()


    def recibirIMG(self):
        if(self.cap.isOpened()):
            ret, frame = self.cap.read() 
            h, w, c = frame.shape
            print('width:  ', w)
            print('height: ', h)
            cv_image = cv2.resize(frame, (int(h*0.5),int(w*0.5)))

        else: 
            self.cap.release()
            print("cap not opened")

        image = cv_image[0:int (len(cv_image)*0.8),int (len(cv_image[0])*0.2): int (len(cv_image[0])*0.8)]
        self.detectar_colores(image)
        figura = self.detectar_figura(image)
        self.detectar_letras(image)
        #print (f"La figura es: {figura}")
        ruta ="/home/sebastian/Uniandes202310/Robotica/proyecto_final/proyecto_final_ws/src/mi_robot_vision/mi_robot_vision/perspectiva_actual.png"
		
        cv2.imshow("Image window", image)
        #cv2.imwrite (ruta,image)
        cv2.waitKey(1)
        tf = time.time()
    
#############################################################################################################
    def detectar_letras(self,image):
        result = self.reader.readtext(image, paragraph=True)
        for res in result:
            print("res:", res[1])
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
    def detectar_colores(self,image):
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
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y),
                                        (x + w, y + h),
                                        (255, 0, 0), 2)
                cv2.putText(image, "Blue Colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 0, 0))
                
def main(args=None):
    rclpy.init(args=args)
    analisis_imagen = Analisis_Imagen()
    rclpy.spin(analisis_imagen)
    analisis_imagen.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
