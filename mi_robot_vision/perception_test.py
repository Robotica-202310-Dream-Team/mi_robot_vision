import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from proyecto_interfaces.srv import StartPerceptionTest
from proyecto_interfaces.msg import Banner
from std_msgs.msg import String, Float32MultiArray, Bool



class Perception_test(Node):

    def __init__(self):
        super().__init__('perception_test')
        self.srv = self.create_service(StartPerceptionTest, 'group_12/start_perception_test_srv', self.read_txt_callback)
        self.publisher = self.create_publisher(Banner, 'vision/banner_group_12', 10)
        self.sub_pos_final= self.create_subscription(Float32MultiArray,'posicion_final',self.callback_pos_final, 10)
        self.sub_llego= self.create_subscription(Bool,'llego',self.callback_llego, 10)

        print('Servicio para la prueba de percepción listo')
        x_goal_1 = 0
        y_goal_1 = 0  
        x_goal_2 = 0
        y_goal_2 = 0

    def read_txt_callback(self, request, response):
        print('Se ha llamado al servicio para la prueba de percepción')
        banner_a_goal = request.banner_a
        banner_b_goal = request.banner_b
        print("Los banners a identificar son: " + str(banner_a_goal) + " y "+ str(banner_b_goal))
        if banner_a_goal == 1:
            x_goal_1 = 0
            y_goal_1 = 0
        elif banner_a_goal == 2:
            x_goal_1 = 0
            y_goal_1 = 0
        elif banner_a_goal == 3:
            x_goal_1 = 0
            y_goal_1 = 0  

        if banner_b_goal == 1:
            x_goal_2 = 0
            y_goal_2 = 0
        elif banner_b_goal == 2:
            x_goal_2 = 0
            y_goal_2 = 0
        elif banner_b_goal == 3:
            x_goal_2 = 0
            y_goal_2 = 0   
        response.answer = "Debo identificar el banner a que se encuentra en las coordenadas ("+str(x_goal_1)+","+str(y_goal_1)+") y el banner b que se encuentra en las coordenadas ("+str(x_goal_2)+","+str(y_goal_2)+")."
        return response



def main(args=None):
    rclpy.init(args=args)
    perception_test = Perception_test()
    rclpy.spin(perception_test)
    perception_test.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()