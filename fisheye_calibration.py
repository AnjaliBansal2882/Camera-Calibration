import cv2
import numpy as np
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# for fisheye camera calibration only(comment otherwise)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1,8*15,3), dtype=np.float32)
objp[0,:,:2] = np.mgrid[0:8,0:15].T.reshape(-1,2)

# arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane.

# Load 20 PNG images from your dataset (modify the path)
images = []
i=0
for i in range(30):
  filename = f"/content/IMX335_640x480_{i+1}.jpg"  # Adjust numbering if needed
  img = cv2.imread(filename)
  if img is not None:
    images.append(img)
  else:
    print(f"Error: Couldn't read image {filename}")
    continue

count = 0
# Process each image for calibration
for img in images:
  count+=1
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  alpha = 2         #to increase contrast
  beta = 15       #to increase brightness
  img2_g = cv2.convertScaleAbs(gray, alpha = alpha, beta=beta)

  # find the chessboard corners
  ret, corners = cv2.findChessboardCorners(img2_g, (8,15), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

  # If found, refine corner positions and add points to object and image point lists
  if ret:
    print(f"entered ret for{count}")
    objpoints.append(objp)
    cv2.cornerSubPix(img2_g, corners, (8,15), (-1,-1), criteria)
    imgpoints.append(corners)

    # Draw and display the corners (optional)
    cv2.drawChessboardCorners(img2_g, (8,15), corners, ret)
    plt.imshow(img2_g)
  else:
    alpha = 3         #to increase contrast
    beta = 15       #to increase brightness
    img2_g = cv2.convertScaleAbs(gray, alpha = alpha, beta=beta)
    # Find corners again with alpha = 3
    ret, corners = cv2.findChessboardCorners(img2_g, (8,15), None)
    if ret:
      print(f"entered ret for{count}")
      cv2.cornerSubPix(img2_g, corners, (8,15), (-1,-1), criteria)
      objpoints.append(objp)
      imgpoints.append(corners)

      # Draw and display the corners (optional)
      cv2.drawChessboardCorners(img2_g, (8,15), corners, ret)
      plt.imshow(img2_g)

    else:
      print(f"Corners not found in image {count}")
    continue


objpoints = [np.asarray(objp, dtype=np.float32) for objp in objpoints]
imgpoints = [np.asarray(imgp, dtype=np.float32) for imgp in imgpoints]

# Calibrate the camera
if len(objpoints) > 0 and len(imgpoints) > 0:
  K = np.zeros((3, 3))
  D = np.zeros((4, 1))
  print(type(imgpoints[0][0]))
  rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(15)]
  tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(15)]
  rms, _, _, _, _ = \
  cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
  #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
else:
  print(len(objpoints))
  print(len(imgpoints))
  print("No valid calibration points found.")


# Print camera calibration parameters
print("Camera matrix:")
print(K)
print("\nDistortion coefficients:")
print(D)
print("\nRotation vectors:")
for rvec in rvecs:
  print(rvec)
print("\nTranslation vectors:")
for tvec in tvecs:
  print(tvec)

# You can now use these parameters for undistortion or other camera operations
