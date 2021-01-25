import argparse
import cv2
import numpy as np
from imutils.video import FPS
import imutils
import time

# Argument input
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to video")
ap.add_argument("-o", "--output", required=True, help="path to output")

args = vars(ap.parse_args())

# Upper, lower boundary in HSV color space
upper_ball = np.array([255, 240, 255])
lower_ball = np.array([200, 60, 120])

# Process argument
print("[INFO] Start to process video")
vs = cv2.VideoCapture(args["video"])  # Initialize the video
# Let system and video warm up
time.sleep(2.0)
# Initialize writer for video output
writer = None

# Initialize position array
position = []

# Get video frame rate
video_frame_rate = vs.get(cv2.CAP_PROP_FPS)  # Get video frame rate
print("[INFO] Quit by pressing q")
while True:
    (grabbed, frame) = vs.read()

    # the end of video
    if frame is None:
        break

    new_frame = frame[150:1030, :1500]
    (H, W) = frame.shape[:2]
    if writer is None:  # Initialize the writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            args["output"], fourcc, video_frame_rate, (W, H), True)

    # Blur the image then convert to HSV color
    blurred = cv2.GaussianBlur(new_frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Detect object of specified color
    mask = cv2.inRange(blurred, lower_ball, upper_ball)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Mask", mask)
    # Refine detection
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contour = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    center = None
    # Calculate center and draw contour, trajectory
    if len(contour) > 0:

        # Chose the smaller contour
        c = max(contour, key=cv2.contourArea)
        #ball is round
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        centerx, centery = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        cv2.circle(frame, (int(x), int(y)+150),
                   int(radius), (255, 0, 0), 2)
        cv2.circle(frame, (centerx, centery+150), 3, (0, 255, 0), -1)
    position.append((centerx, centery+150))

    for i in range(len(position)-1):
        cv2.line(frame, position[i], position[i+1], (0, 255, 0), 3)

    # Fit image display with screen
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # Display
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    writer.write(frame)
    # quit by pressing "q"
    if key == ord("s"):
        break

if writer:
    writer.release()
vs.release()
cv2.destroyAllWindows()
print("Done")
