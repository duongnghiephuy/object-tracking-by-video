# Object tracking by video

Simple execution with OpenCV 3.x 

## Single object tracking by videos by ROI

Trackers: 

    "csrt": cv2.TrackerCSRT_create
    
    "kcf": cv2.TrackerKCF_create
    
    "boosting": cv2.TrackerBoosting_create
    
    "mil": cv2.TrackerMIL_create
    
    "tld": cv2.TrackerTLD_create
    
    "medianflow": cv2.TrackerMedianFlow_create
    
    "mosse": cv2.TrackerMOSSE_create

default is "csrt"

Example command: `-python roi track.py --video videos/test0.mp4 --output output/test0.avi --tracker csrt`

Press "s" to stop the video and choose object to track.
Press "Enter" to resume.\


## Single object tracking by YOLOv3 deep learning detection and OpenCV tracker.

YOLOv3 was trained by Darknet team on COCO dataset. https://pjreddie.com/darknet/yolo/

It consists of 3 files: coco.names cotaining all labels, yolov3.config, and yolov3.weights.

Default tracker is kcf.

Command: `-python yolo_object_tracker.py --video videos/cat1.mp4 --yolo yolo-coco --output output/cat1.avi --label cat`


![ezgif com-gif-maker](https://user-images.githubusercontent.com/55075721/140869154-2608efb0-f600-46ea-85b1-950b9d6b72cf.gif)



## Single object tracking by SSD deep learning detection and OpenCV tracker

SSD is more precise than YOLOv3.
SSD caffe model was trained by weiliu89 on COCO dataset https://github.com/weiliu89/caffe/tree/ssd
It consists of 3 files: coco.names cotaining all labels, deploy.prototxt, and VGG_coco_SSD.caffemodel

Command: `-python ssd_object_tracker.py --video videos/dog.mp4 --ssd ssd --output output/dog.avi --label dog --tracker csrt`

![ezgif com-gif-maker](https://user-images.githubusercontent.com/55075721/140869268-b0b9d244-c400-4eab-bbbf-06a0c603e9a0.gif)





## Single object tracking by color 

Specify lower and upper color in HSV color space then run. It's important to observe bit mask result of the specified color range. 
Objects of the same color which is not of interest can be cut out by shaping the image.

## Multiple objects tracking: trying 
## Tracking distance during COVID: trying 



