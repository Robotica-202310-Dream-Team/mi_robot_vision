#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")

fswebcam -r 1280.0x720.0 --no-banner /home/robotica/proyecto_ws/src/mi_robot_vision/mi_robot_vision/$DATE.jpg
