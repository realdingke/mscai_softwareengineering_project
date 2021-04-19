# !pip install opencv-contrib-python==4.2.0.34

import time
import cv2
import os
import sys
import argparse

# USAGE
# python3 OpenCV_tracker.py --video /Users/apple/2seconds.mp4 --tracker MOSSE -gt /Users/apple/gt.txt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="CSRT",
                help="OpenCV object tracker type: BOOSTING, MIL,\
                KCF,TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT")
ap.add_argument("-gt","--ground_truth_file", type=str,
                help="OpenCV object tracker type")
args = ap.parse_args()


time_start=time.time()

fairmot_tracking_result = open(args.ground_truth_file)
lines = fairmot_tracking_result.readlines()
tracking_info = []
for line in lines:
    tracking_info.append(line.split(','))

object_id = []
first_appearance = []
for result in tracking_info:
    if int(float(result[1])) not in object_id:
        object_id.append(int(float(result[1])))
        first_appearance.append(result)

tracking_result = []

for i in range(len(first_appearance)):
    tracking_result.append([int(float(first_appearance[i][0])),
                            int(float(first_appearance[i][1])),
                            float(first_appearance[i][2]),
                            float(first_appearance[i][3]),
                            float(first_appearance[i][4]),
                            float(first_appearance[i][5])])

f = open(f"{args.tracker}.txt", "w")

# version of OpenCV
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')       
if int(major_ver) < 4 and int(minor_ver) < 3:
    tracker_class = cv2.cv2.Tracker_create(args.tracker)
else:
    if args.tracker == 'MOSSE':
        tracker_class = cv2.TrackerMOSSE_create
    elif args.tracker == "CSRT":
        tracker_class = cv2.TrackerCSRT_create
    elif args.tracker == 'KCF':
        tracker_class = cv2.TrackerKCF_create
    elif args.tracker == 'TLD':
        tracker_class = cv2.TrackerTLD_create
    elif args.tracker == 'MEDIANFLOW':
        tracker_class = cv2.TrackerMedianFlow_create
    elif args.tracker == 'GOTURN':
        tracker_class = cv2.TrackerGOTURN_create
    elif args.tracker == 'BOOSTING':
        tracker_class = cv2.TrackerBoosting_create
    elif args.tracker == 'MIL':
        tracker_class = cv2.TrackerMIL_create
    else:
        tracker = None
        print('Incorrect tracker name')

video = cv2.VideoCapture(args.video)
width = video.get(3)
height = video.get(4)

current_path = os.getcwd()
os.mkdir(current_path+f'/{args.tracker}')

total_frame_nb = video.get(7)

if not video.isOpened():
    print("Could not open video")
    sys.exit()    


def tracker(i):
    return cv2.MultiTracker_create() 
tracker_list = []
for i in range(len(first_appearance)):
    tracker_list.append(tracker(1))


frame_nb = 0

fail_obj = []
# loop over frames from the video stream
while True: 
    frame_nb += 1
    
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    ok, frame = video.read() 
    
    # check to see if we have reached the end of the stream
    if ok is None or frame is None:
        break
        
    for index,info in enumerate(first_appearance):
        if int(float(info[0]))==frame_nb:
            box = (int(float(info[2])),int(float(info[3])),
                   int(float(info[4])),int(float(info[5])))
            tracker = tracker_class()
            # allocate each object a tracker
            tracker_list[index].add(tracker, frame, box) 

    # grab the updated bounding box coordinates for each object that is being tracked
    S = []
    Box = []
    for index, info in enumerate(tracker_list):
        if int(float(first_appearance[index][0]))<=frame_nb:
            success, boxes = info.update(frame) 
            S.append(success)
            Box.append(boxes)

    # loop over the bounding boxes and draw then on the frame
    for obj in range(len(S)): # loop over each object's tracker
        if int(float(first_appearance[obj][0]))<=frame_nb:
            if S[obj]==True and\
                obj not in fail_obj and\
                  (Box[obj][0][1]+Box[obj][0][3])<height and\
                  (Box[obj][0][0]+Box[obj][0][2])<width and\
                    (Box[obj][0][1])>0 and\
                    (Box[obj][0][0]>0) and\
                      (frame_nb-int(float(first_appearance[obj][0])))<130:   
                # Tracking success
                p1 = (int(Box[obj][0][0]), int(Box[obj][0][1]))
                p2 = (int(Box[obj][0][0] + Box[obj][0][2]),
                      int(Box[obj][0][1] + Box[obj][0][3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, f"{obj+1}", (int(Box[obj][0][0]), int(Box[obj][0][1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                tracking_result.append([frame_nb+1, obj+1, Box[obj][0][0], Box[obj][0][1],
                                        Box[obj][0][2], Box[obj][0][3]])
            else:      
                fail_obj.append(obj)
                tracker_list[obj]=cv2.MultiTracker_create() 
                
    # save output images
    frame_name = "image" +str(frame_nb) +".jpg"
    cv2.imwrite(current_path+f'/{args.tracker}/{frame_name}',frame,\
                [cv2.IMWRITE_JPEG_QUALITY,100])

    key = cv2.waitKey(1) & 0xFF  

# Stop when the video is finished
video.release()
time_end=time.time()
print('time cost',time_end-time_start,'s')

# save video
img_array = []
for i in range(int(frame_nb)-1):
    filename = current_path+f'/{args.tracker}/image{i+1}.jpg'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter(f'tracking_result_{args.tracker}.avi',\
                      cv2.VideoWriter_fourcc(*'MJPG'), 24, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()    # Release the video recording


# close all windows
cv2.destroyAllWindows()


# sort the tracking result and save as txt
tracking_result.sort()
for i in range(len(tracking_result)):
    f.write(f'{tracking_result[i][0]},{tracking_result[i][1]},\
    {tracking_result[i][2]},{tracking_result[i][3]},\
    {tracking_result[i][4]},{tracking_result[i][5]}\n')

f.close()




