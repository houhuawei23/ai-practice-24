import cv2
import dlib
import sys
import os
import numpy as np

data_path = "./imgs"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


def cal_feature(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detector(img_gray, 1)
    shape = predictor(img_gray, face[0])
    face_descriptor = face_model.compute_face_descriptor(img, shape)
    return face_descriptor


def cal_feature_distance(feature1, feature2):
    npfeature1 = np.array(feature1)
    npfeature2 = np.array(feature2)
    return np.linalg.norm(npfeature1 - npfeature2)


def face_capture():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    while True:
        readsuccess, img = camera.read()
        if readsuccess:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_img, 1)
            if len(faces) > 1:
                print("More than one face detected!")
            elif len(faces) == 0:
                print("No face detected!")
            else:
                camera.release()
                return cal_feature(img)


def face_identification():
    now_feature = face_capture()

    for entry in os.scandir(data_path):
        if entry.is_dir:
            name = entry.name
            for dirpath, dirnames, filenames in os.walk(entry):
                for filename in filenames:
                    if filename.endswith(".jpg"):
                        img_path = os.path.join(dirpath, filename)
                        img = cv2.imread(img_path)
                        feature = cal_feature(img)
                        distance = cal_feature_distance(now_feature, feature)
                        if distance < 0.6:
                            print("The person is: ", name)
                            return
    print("Face not found")


if __name__ == "__main__":
    face_identification()