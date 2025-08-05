import cv2
import numpy as np
import matplotlib.pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((4*4,3), dtype=np.float32)
objp[:,:2] = np.mgrid[0:4,0:4].T.reshape(-1,2)

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane.

images = []
i=0
for i in range(19):
  filename = f"/content/pi_pic{i+1}.jpg"  # Adjust numbering if needed
  img = cv2.imread(filename)
  if img is not None:
    images.append(img)
  else:
    print(f"Error: Couldn't read image {filename}")
    continue

count = 0
for img in images:
  count+=1
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # the mathematical formula is
  # new(i,j) = alpha * old(i,j) + beta
  alpha = 3         #to increase contrast
  beta = 15         #to increase brightness
  img2_g = cv2.convertScaleAbs(gray, alpha = alpha, beta=beta)

  # finding the chessboard corners
  ret, corners = cv2.findChessboardCorners(img2_g, (4,4), None)

  if ret:
    print(f"entered ret for{count}")
    cv2.cornerSubPix(img2_g, corners, (4,4), (-1,-1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners.reshape(-1,2))

    # Draw and display the corners
    cv2.drawChessboardCorners(img2_g, (4,4), corners, ret)
    plt.imshow(img2_g)


if len(objpoints) > 0 and len(imgpoints) > 0:
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
else:
  print(len(objpoints))
  print(len(imgpoints))
  print("No valid calibration points found.")


print("Camera matrix:")
print(mtx)
print("\nDistortion coefficients:")
print(dist)
print("\nRotation vectors:")
for rvec in rvecs:
  print(rvec)
print("\nTranslation vectors:")
for tvec in tvecs:
  print(tvec)
