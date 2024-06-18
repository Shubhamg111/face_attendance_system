# import cv2
# import pickle
# import numpy as np
# import os

# # Ensure the data directory exists
# data_dir = 'data'
# if not os.path.exists(data_dir):
#     os.makedirs(data_dir)

# # Load face detector
# facedetect = cv2.CascadeClassifier(os.path.join(data_dir, 'haarcascade_frontalface_default.xml'))

# # Initialize variables
# facesData = []
# i = 0

# name = input("Enter your name: ")

# # Capture video
# video = cv2.VideoCapture(0)

# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("Failed to capture image")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         cropImg = frame[y:y+h, x:x+w, :]
#         resizedImg = cv2.resize(cropImg, (50, 50))
#         if len(facesData) <= 100 and i % 10 == 0:
#             facesData.append(resizedImg)
#         i += 1
#         cv2.putText(frame, str(len(facesData)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 225), 1)

#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
#     if k == ord('q') or len(facesData) == 100:
#         break

# video.release()
# cv2.destroyAllWindows()

# facesData = np.asarray(facesData)
# facesData = facesData.reshape(100, -1)

# # Load or initialize the names and faces data
# names_file = os.path.join(data_dir, 'names.pkl')
# faces_file = os.path.join(data_dir, 'facesData.pkl')

# if not os.path.exists(names_file):
#     names = [name] * 100
#     with open(names_file, 'wb') as f:
#         pickle.dump(names, f)
# else:
#     with open(names_file, 'rb') as f:
#         names = pickle.load(f)
#     names += [name] * 100
#     with open(names_file, 'wb') as f:
#         pickle.dump(names, f)

# if not os.path.exists(faces_file):
#     with open(faces_file, 'wb') as f:
#         pickle.dump(facesData, f)
# else:
#     with open(faces_file, 'rb') as f:
#         faces = pickle.load(f)
#     faces = np.append(faces, facesData, axis=0)
#     with open(faces_file, 'wb') as f:
#         pickle.dump(faces, f)



import cv2
import pickle
import numpy as np
import os

# Ensure the data directory exists
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Load the pre-trained model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Initialize variables
facesData = []
i = 0

name = input("Enter your name: ")

# Capture video
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
            resizedImg = cv2.resize(cropImg, (50, 50))
            if len(facesData) <= 100 and i % 10 == 0:
                facesData.append(resizedImg)
            i += 1
            cv2.putText(frame, str(len(facesData)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (50, 50, 225), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(facesData) == 100:
        break

video.release()
cv2.destroyAllWindows()

facesData = np.asarray(facesData)
facesData = facesData.reshape(100, -1)

# Load or initialize the names and faces data
names_file = os.path.join(data_dir, 'names.pkl')
faces_file = os.path.join(data_dir, 'facesData.pkl')

if not os.path.exists(names_file):
    names = [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

if not os.path.exists(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(facesData, f)
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, facesData, axis=0)
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)

