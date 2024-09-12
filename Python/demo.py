import torch
import cv2
import pathlib
import time
import mysql.connector
from facenet_pytorch import MTCNN
from scipy.spatial import distance as dist
from collections import OrderedDict, defaultdict
from model import model_static
import numpy as np
from PIL import Image
from torchvision import transforms
import uuid  


# Define the CentroidTracker class (same as above)

class CentroidTracker:
    def __init__(self, initObject = 0, maxDisappeared=50):
        self.nextObjectID = initObject
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.bboxes = OrderedDict()  # To store the bounding boxes

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bboxes[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.bboxes

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
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
                self.bboxes[objectID] = rects[col]
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
                    self.register(inputCentroids[col], rects[col])

        return self.objects, self.bboxes

# Set up database connection
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="eyettention"
)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_detections (
        id INT AUTO_INCREMENT PRIMARY KEY,
        unique_id VARCHAR(36) UNIQUE,
        timestamp DATETIME,
        x1 INT,
        y1 INT,
        x2 INT,
        y2 INT,
        looking BOOLEAN
    )
''')
conn.commit()

# Initialize variables
CWD = pathlib.Path.cwd()
cap = cv2.VideoCapture(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device="cuda")
model = model_static("model_weights.pkl")
model_dict = model.state_dict()
snapshot = torch.load("model_weights.pkl", map_location="cuda")
model_dict.update(snapshot)
model.load_state_dict(model_dict)
model.eval()

# Set up stream key (replace with your actual stream key)
rtmp_url = "rtmp://a.rtmp.youtube.com/live2/9vf2-8tuz-d5wm-r1b1-396u"

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cursor = conn.cursor()
query = 'SELECT id FROM face_detections ORDER BY timestamp DESC LIMIT 1'
cursor.execute(query)
result = cursor.fetchall()


integer_value = None
print(len(result))
if len(result) == 0:
    integer_value = 0
else:
    resultTup = result[0]
    integer_value =  resultTup[0]


print(integer_value)

ct = CentroidTracker(initObject = integer_value + 1, maxDisappeared=50)
trackableObjects = defaultdict(lambda: {'start_time': None, 'looking': False})


with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _ = mtcnn.detect(frame)

        current_time = time.time()

        rects = []
        if boxes is not None and len(boxes) != 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                box_width = x2 - x1
                box_height = y2 - y1
                new_x1 = max(0, x1 - int(0.2 * box_width))
                new_y1 = max(0, y1 - int(0.2 * box_height))
                new_x2 = min(frame.shape[1], x2 + int(0.2 * box_width))
                new_y2 = min(frame.shape[0], y2 + int(0.4 * box_height))  # Increase more height for neck and shoulders

                rects.append((new_x1, new_y1, new_x2, new_y2))

        objects, bboxes = ct.update(rects)
        
        looking_count = 0
        not_looking_count = 0

        for (objectID, centroid) in objects.items():
            new_x1, new_y1, new_x2, new_y2 = bboxes[objectID]
            face = frame[new_y1:new_y2, new_x1:new_x2]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = preprocess(face_pil)
            face_batch = face_tensor.unsqueeze(0)

            output = model(face_batch)
            output = torch.mean(output, 0)
            score = torch.sigmoid(output).item()
            looking = score > 0.5

            if trackableObjects[objectID]['start_time'] is None:
                trackableObjects[objectID]['start_time'] = current_time

            unique_id = str(uuid.uuid4())  # Generate a unique ID for each detection

            if looking:
                looking_count += 1
                if not trackableObjects[objectID]['looking']:
                    trackableObjects[objectID]['start_time'] = current_time
                    trackableObjects[objectID]['looking'] = True
                elif current_time - trackableObjects[objectID]['start_time'] >= 5:
                    try:
                        cursor.execute('''
                            INSERT INTO face_detections (id, unique_id, timestamp, x1, y1, x2, y2, looking)
                            VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                            timestamp=NOW(), x1=VALUES(x1), y1=VALUES(y1), x2=VALUES(x2), y2=VALUES(y2), looking=VALUES(looking)
                        ''',
                                       (objectID, unique_id, new_x1, new_y1, new_x2, new_y2, True))
                        conn.commit()
                    except mysql.connector.Error as err:
                        print(f"Error: {err}")
            else:
                not_looking_count += 1
                if trackableObjects[objectID]['looking']:
                    try:
                        cursor.execute("INSERT INTO face_detections (id, unique_id, timestamp, x1, y1, x2, y2, looking) VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s)",
                                       (objectID, unique_id, new_x1, new_y1, new_x2, new_y2, False))
                        conn.commit()
                    except mysql.connector.Error as err:
                        print(f"Error: {err}")
                trackableObjects[objectID]['looking'] = False

            predicted_class_name = "looking" if looking else "Not Looking"
            cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 3)
            cv2.putText(frame, f"ID {objectID}: {predicted_class_name}", (new_x1, new_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        total_count = looking_count + not_looking_count
        legend_text = f"Looking: {looking_count}  Not Looking: {not_looking_count}  Total: {total_count}"
        cv2.putText(frame, legend_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
        
        cv2.imshow("NITEC", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
conn.close()
