# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime
# from win32com.client import Dispatch

# def speak(str1):
#     speak = Dispatch("SAPI.SpVoice")
#     speak.Speak(str1)

# # Ensure the data directory exists
# data_dir = 'data'
# attendance_dir = 'Attendance'
# if not os.path.exists(attendance_dir):
#     os.makedirs(attendance_dir)

# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier(os.path.join(data_dir, 'haarcascade_frontalface_default.xml'))

# with open(os.path.join(data_dir, 'names.pkl'), 'rb') as f:
#     LABELS = pickle.load(f)

# with open(os.path.join(data_dir, 'facesData.pkl'), 'rb') as f:
#     FACES = pickle.load(f)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# COL_NAMES = ['Name', 'Time']

# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("Failed to capture image")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         cropImg = frame[y:y+h, x:x+w, :]
#         resizedImg = cv2.resize(cropImg, (50, 50)).flatten().reshape(1, -1)
#         output = knn.predict(resizedImg)
#         ts = time.time()
#         date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
#         csv_file_path = os.path.join(attendance_dir, "Attendance_" + date + ".csv")
#         exist = os.path.isfile(csv_file_path)

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
#         cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
#         cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

#         attendance = [str(output[0]), str(timestamp)]
      
#         k = cv2.waitKey(1)
#         if k == ord('o'):
#             speak("Attendance Taken..")
#             time.sleep(5)
#             if exist:
#                 with open(csv_file_path, 'r') as f:
#                     existing_entries = [row for row in csv.reader(f)]
#                     if any(row[0] == attendance[0] for row in existing_entries):
#                         continue

#             with open(csv_file_path, 'a', newline='') as csvfile:
#                 writer = csv.writer(csvfile)
#                 if not exist:
#                     writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)

#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()


from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

# Ensure the data directories exist
data_dir = 'data'
attendance_dir = 'Attendance'
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

# Load the pre-trained face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load the names and face data
with open(os.path.join(data_dir, 'names.pkl'), 'rb') as f:
    LABELS = pickle.load(f)

with open(os.path.join(data_dir, 'facesData.pkl'), 'rb') as f:
    FACES = pickle.load(f)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Column names for CSV
COL_NAMES = ['Name', 'Time']

# Start video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture image")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cropImg = frame[startY:endY, startX:endX, :]
            resizedImg = cv2.resize(cropImg, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resizedImg)

            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            csv_file_path = os.path.join(attendance_dir, "Attendance_" + date + ".csv")
            exist = os.path.isfile(csv_file_path)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (50, 50, 255), 2)
            cv2.rectangle(frame, (startX, startY-40), (endX, startY), (50, 50, 255), -1)
            cv2.putText(frame, str(output[0]), (startX, startY-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            attendance = [str(output[0]), str(timestamp)]
      
            k = cv2.waitKey(1)
            if k == ord('o'):
                speak("Attendance Taken..")
                time.sleep(5)
                if exist:
                    with open(csv_file_path, 'r') as f:
                        existing_entries = [row for row in csv.reader(f)]
                        if any(row[0] == attendance[0] for row in existing_entries):
                            continue

                with open(csv_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not exist:
                        writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('Q'):
        break

video.release()
cv2.destroyAllWindows()

