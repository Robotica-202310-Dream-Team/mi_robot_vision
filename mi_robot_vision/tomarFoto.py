#DATE=$(date +"%Y-%m-%d_%H%M")
#fswebcam -r 1280.0x720.0 --no-banner /home/robotica/proyecto_ws/src/mi_robot_vision/mi_robot_vision/$DATE.jpg
#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import cv2
import os
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImagePublisher(Node):
    def __init__(self):
        super().__init__("image_publisher")
        print("Inicio del nodo que publica la imagen")
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        self.pub = self.create_publisher(Image, "/video", 10)

    def run(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read() 
            if ret:
                self.pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    ip = ImagePublisher()
    print("Publishing...")
    ip.run()

    ip.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()