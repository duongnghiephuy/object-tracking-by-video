# Object tracking by video

OpenCV 3.x 

Single object tracking by videos by ROI

Trackers: 

    "csrt": cv2.TrackerCSRT_create
    
    "kcf": cv2.TrackerKCF_create
    
    "boosting": cv2.TrackerBoosting_create
    
    "mil": cv2.TrackerMIL_create
    
    "tld": cv2.TrackerTLD_create
    
    "medianflow": cv2.TrackerMedianFlow_create
    
    "mosse": cv2.TrackerMOSSE_create

default is "csrt"

Example command: -python roi track.py --video videos/test0.mp4 --output output/test0.avi --tracker csrt

Press "s" to stop the video and choose object to track.
Press "Enter" to resume.

Object tracking by YOLOv3 deep learning detection and OpenCV tracker.

YOLOv3 were trained by Darknet team on COCO dataset. https://pjreddie.com/darknet/yolo/
It consists of 3 files: coco.names cotaining all labels, yolov3.config, and yolov3.weights.
Default tracker is kcf.

Command: -python yolo_object_tracker.py --video videos/cat1.mp4 --output output/cat1.avi --label cat 










