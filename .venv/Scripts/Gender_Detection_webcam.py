import cv2
import numpy as np

# Load models
face_model = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
gender_model = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['Male', 'Female']

# Start webcam
cap = cv2.VideoCapture(0)

# Set capture resolution (optional, improves output clarity)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()
    h, w = frame.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=True)
    face_model.setInput(blob)
    detections = face_model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            face = frame[max(0, y1 - 15):min(y2 + 15, h - 1),
                         max(0, x1 - 15):min(x2 + 15, w - 1)]

            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                              MODEL_MEAN_VALUES, swapRB=True)
            gender_model.setInput(face_blob)
            preds = gender_model.forward()
            gender = gender_list[preds[0].argmax()]
            confidence_percent = round(preds[0].max() * 100, 2)

            label = f"{gender}"
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    #print("The confidence % is ", {confidence_percent})


    output_frame = cv2.resize(frame_copy, (350,270))  # Resize to larger dimensions
    cv2.imshow("Gender Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
