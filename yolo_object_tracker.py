import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from imutils.video import FPS
import imutils

# Command line Argument
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to video")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to yolo directory")
ap.add_argument("-c", "--confidence",
                help="minium confidence to determine label", type=float, default=0.04)
ap.add_argument("-t", "--tracker", type=str,
                default="kcf", help="OpenCV tracker type")
ap.add_argument("-l", "--label", help="Label to track", required=True)
ap.add_argument("-o", "--output", required=True, help="path to output")
args = vars(ap.parse_args())

# Path to label file
labelspath = Path(args["yolo"])/"coco.names"
LABELS = open(labelspath).read().strip().split("\n")

# Path to yolo files
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Input yolo model to openCV
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

print("[INFO] Start to process video")
# Read video
vs = cv2.VideoCapture(args["video"])
video_frame_rate = vs.get(cv2.CAP_PROP_FPS)  # Get video frame rate
writer = None
label = " "
fps = FPS().start()

# Initialize openCV trackers
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

box = []
print("[INFO] Initialization")

while True:
    # Read frame by frame
    (grabbed, frame) = vs.read()
    # End if there is no frame
    if frame is None:
        break
    (H, W) = frame.shape[:2]

    # initialize video writer
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            args["output"], fourcc, video_frame_rate, (W, H), True)
    # Yolo detection will run until the target label is detected

    if not box:

        # Generate a blob. The shape and scale here depends on the model
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, size=(416, 416), swapRB=True, crop=False)

        # Input blob to model
        net.setInput(blob)

        # Take the output layers' names
        ln = net.getLayerNames()
        ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

        # Take output layers by output layers' names
        layerOutputs = net.forward(ln)
        max_confidence = 0
        for output in layerOutputs:
            for detection in output:
                # In each detection of each output layer, we take the detection with highest objectness
                scores = detection[5:]
                classID = np.argmax(scores)
                if LABELS[classID] == args["label"]:
                    confidence = scores[classID]

                    # confidence threshold
                    if confidence > args["confidence"] and confidence > max_confidence:
                        max_confidence = confidence
                        box = detection[:4]*(W, H, W, H)
                        centerX, centerY, width, height = box.astype("int")
                        X = int(centerX-width/2)
                        Y = int(centerY-height/2)
                        box = [X, Y, int(width), int(height)]

        print("[INFO] Detected:", box)
        if box:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, args["label"], (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            tracker.init(frame, (x, y, w, h))
            fps = FPS().start()
    else:
        # Once object is detected, tracker tracks object
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Count frame rate of tracker
        fps.update()
        fps.stop()

        info = [("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"), ("FPS:", "{:.2f}".format(fps.fps()))]
        for (i, (k, v)) in enumerate(info):
            text = "{}:{}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if writer:
        writer.write(frame)

print("[INFO]:Writing")
if writer:
    writer.release()
vs.release()
print('Done writing')
cv2.destroyAllWindows()
