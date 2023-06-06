#Primer archivo prueba de reconocimiento de imagenes
import cv2

image = cv2.imread('prueba1.jpeg')
def detectar_figura(image): 
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
                figure="Ciruculo"
                cv2.putText(image,figure, (x,y-5),1,1.5,(0,255,0),2)
        n +=1
    
    cv2.drawContours(image, [approx], 0, (0,255,0),2)
    #cv2.imshow('image',image)
    #cv2.waitKey(0)    
    return figure

print (f"la figura es: un {detectar_figura(image)}")