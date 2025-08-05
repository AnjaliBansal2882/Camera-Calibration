# Camera-Calibration

Camera calibration is the process of determining the intrisic and extrinisic properties of a camera. The intrinsic properties of a camera include its focal length and principal point in the x and y directions. The extricnsic properties of a camera includes computing its rotation and translation matrix. The intrinsic properties of a camera remain constant whereas the extrensic properties change with its orientation. 

This repo contains python scripts that compute the camera's intrinsics properties based on its field of view using the Checkerboard method. I utilized the OpenCV library for the task. 
The flow of the script is as follows: 
- It takes 20 checkerboard images, taken from different angles and height as input(be sure to capture the entire chessboard in each image but avoid extra area in image)
- detects corners in each image
- convers the 3D coordinates to 2D plane
- calls the calibratecamera() function from which outpputs the intrinsic matrix, rotation vector and the translation vector. 
