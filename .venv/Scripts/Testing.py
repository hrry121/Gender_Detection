import cv2
import os
import numpy as np

# Paths to gender model
gen1 = "gender_deploy.prototxt"
gen2 = "gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

# Load model
genderNet = cv2.dnn.readNet(gen2, gen1)

# Directory structure
test_dir = "C:/Users/Admin/Documents/Gender_Test"
correct = 0
total = 0

for gender_folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, gender_folder)
    if not os.path.isdir(folder_path):
        continue
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize and blob
        img = cv2.resize(img, (227, 227))
        blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
        genderNet.setInput(blob)
        preds = genderNet.forward()
        pred_gender = genderList[preds[0].argmax()]

        # Compare prediction to actual label
        if pred_gender.lower() == gender_folder.lower():
            correct += 1
        total += 1

accuracy = correct / total * 100 if total > 0 else 0
print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
