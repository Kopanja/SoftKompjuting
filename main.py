import numpy as np
from cv2 import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict
from imutils.object_detection import non_max_suppression

class CentroidTracker():
	
    def __init__(self, maxDisappeared=50):

        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
    
    def register(self, centroid):

        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):

        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):

        if len(rects) == 0:

            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
	
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:

                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects



def find_plato_line(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 150, 250)
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 175, minLineLength=600, maxLineGap=100)
    line = None
    
    for line in lines:
        line = line[0]
    return line

def main_programm(bg_hist, bg_tresh, kernel_size, contour_area, non_max_trash, maxDisappeard):
    res_list = []
    for vid_index in range(1,11):
        
        vid_name = "video" + str(vid_index) + ".mp4"
        ct = CentroidTracker(maxDisappeard)
        frame_cnt = 0
        people_cnt = 0
        line = []
        counted_people_set = set()
        backSub = cv2.createBackgroundSubtractorMOG2(bg_hist, bg_tresh,False)
        capture = cv2.VideoCapture(vid_name)

        if not capture.isOpened():
            print('Unable to open:')
            exit(0)
        while True:
            _ret, frame = capture.read()
            
            frame_cnt = frame_cnt + 1
            if frame is None:
                break
            frame_copy = frame.copy()
            if frame_cnt == 1:
                line = find_plato_line(frame_copy)
                x1, y1, x2, y2 = line
                
            rects = []
            
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            frame_copy = cv2.equalizeHist(frame_copy)

            fgMask = backSub.apply(frame_copy)
            kernel = np.ones((3,3),np.uint8)
            kernel_2 = np.ones((5,5),np.uint8)
            fgMask = cv2.dilate(fgMask,kernel)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_2)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
            fgMask = cv2.erode(fgMask, kernel)
                
            (contours, _hierarchy) = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            
            for c in contours:
                if cv2.contourArea(c) < contour_area:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                rects.append((x,y,w,h))	
            
        
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=non_max_trash)
            objects = ct.update(pick)
            
            for(objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)
                if(centroid[1] > line[1] and objectID not in counted_people_set):
                    people_cnt = people_cnt + 1
                    counted_people_set.add(objectID)
                
            cv2.imshow('foreground and background',fgMask)
            cv2.imshow('rgb',frame)
            if cv2.waitKey(7) & 0xFF == ord("q"):
                break
        print(vid_name + " |prebrojano: " + str(people_cnt))
        res_list.append((vid_name, people_cnt))
    return res_list

def test_func(lista_rezultata, bg_hist,bg_trash, kernel_size, contoure_area,non_max_trash):
    sum_cnt = 0
    for index, obj in enumerate(lista_rezultata):
        sum_cnt = sum_cnt + abs(obj[1] - tacna_lista[index])

    ame = float(sum_cnt)/10

    print("AME : " + str(ame) + "\n")

    
    
    

tacna_lista = [4,24,17,23,17,27,29,22,10,23]
lista_rezultata = main_programm(5000,50.0,3,80,0.25,20)
test_func(lista_rezultata,550,50.0, 3, 100,0.2)




