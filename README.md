# Object tracking by video

OpenCV 3.x 

Single object tracking by videos

Trackers: 

    "csrt": cv2.TrackerCSRT_create
    
    "kcf": cv2.TrackerKCF_create
    
    "boosting": cv2.TrackerBoosting_create
    
    "mil": cv2.TrackerMIL_create
    
    "tld": cv2.TrackerTLD_create
    
    "medianflow": cv2.TrackerMedianFlow_create
    
    "mosse": cv2.TrackerMOSSE_create
default is "csrt"

Example command: roi track.py --video videos/test0.mp4 --output output/test0.avi --tracker csrt

Press "s" to stop the video and choose your object to track by the rectangle.
Press "Enter" to resume

