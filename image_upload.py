import cv2
import os
from src.face_detector import get_faces
from src.gender_age_predictor import predict_gender_age

IMAGE_PATH = input("Enter image path: ").strip()

if not os.path.exists(IMAGE_PATH):
    print("❌ Image not found")
    exit(1)

img = cv2.imread(IMAGE_PATH)
faces = get_faces(img)

for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    gender, age = predict_gender_age(face)

    label = f"{gender}, Age: {age}"
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(img, label, (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imshow("Image Upload Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
