import cv2
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img", type=str, required=True,
	help="path to input directory of images to count")

args = vars(ap.parse_args())

image = cv2.imread(args["img"])


output = image.copy()
# image = cv2.GaussianBlur(image,(5,5),0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2.7, 12,100,100,minRadius=13,maxRadius=22)
# ensure at least some circles were found
cnt=0
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		print(r)
		cnt+=1
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	# show the output image
	print(cnt)
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)