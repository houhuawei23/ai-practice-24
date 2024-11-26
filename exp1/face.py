import cv2
import dlib
import sys
import os
import numpy as np
from log import log_event


# Path to the face library
# FACE_LIB_PATH = "./face_library/"
LOG_FILE = "./access_logs.txt"


# Load known faces and their encodings
def load_face_library(face_lib_path, detector, shape_predictor, face_recognizer):
    face_library = {}
    # for file in os.listdir(face_lib_path):
    #     if file.endswith(".jpg") or file.endswith(".png"):
    #         img = cv2.imread(os.path.join(face_lib_path, file))
    #         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         dets = detector(img_rgb)
    #         if len(dets) > 0:
    #             shape = shape_predictor(img_rgb, dets[0])
    #             face_descriptor = np.array(
    #                 face_recognizer.compute_face_descriptor(img_rgb, shape)
    #             )
    #             face_library[file.split(".")[0]] = face_descriptor

    for entry in os.scandir(face_lib_path):
        if entry.is_dir():
            name = entry.name
            for dirpath, dirnames, filenames in os.walk(entry):
                for file in filenames:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        img = cv2.imread(os.path.join(dirpath, file))
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        dets = detector(img_rgb)
                        if len(dets) > 0:
                            shape = shape_predictor(img_rgb, dets[0])
                            face_descriptor = np.array(
                                face_recognizer.compute_face_descriptor(img_rgb, shape)
                            )
                            face_library[name] = face_descriptor

    return face_library


# face_library = load_face_library()


# Compare a detected face with the library
def compare_face(face_descriptor, face_library, threshold=0.35):
    for name, lib_descriptor in face_library.items():
        distance = np.linalg.norm(face_descriptor - lib_descriptor)
        if distance < threshold:
            return name, distance
    return None, None


# Initialize dlib's face detector and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(
    "./data/shape_predictor_68_face_landmarks.dat"
)


def is_smiling(landmarks):
    """
    Check if the person is smiling based on the landmarks.
    A smile can be detected by the distance between the corners of the mouth and the height of the upper lip.
    """
    left_mouth = landmarks[48]
    right_mouth = landmarks[54]
    top_lip = landmarks[51]
    bottom_lip = landmarks[57]

    # Calculate distances (Euclidean)
    horizontal_dist = (
        (right_mouth[0] - left_mouth[0]) ** 2 + (right_mouth[1] - left_mouth[1]) ** 2
    ) ** 0.5
    vertical_dist = (
        (bottom_lip[1] - top_lip[1]) ** 2 + (bottom_lip[0] - top_lip[0]) ** 2
    ) ** 0.5

    smile_ratio = horizontal_dist / vertical_dist
    return smile_ratio > 2.5  # A threshold; adjust based on testing


def check_and_draw_smile(frame, gray_frame, face):
    # Detect facial landmarks
    landmarks = landmark_predictor(gray_frame, face)
    landmarks = [(p.x, p.y) for p in landmarks.parts()]

    # Draw the detected face and landmarks
    for x, y in landmarks:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Check for smiling
    if is_smiling(landmarks):
        cv2.putText(
            frame,
            "Smile Detected!",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            frame,
            "Please Smile",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )


def detect_and_draw_faces(
    frame, detector, shape_predictor, face_recognizer, face_library
):
    # Load the detector and predictor
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray)

    for det in dets:
        check_and_draw_smile(frame, gray, det)
        shape = shape_predictor(frame, det)
        face_descriptor = np.array(
            face_recognizer.compute_face_descriptor(frame, shape)
        )
        name, distance = compare_face(face_descriptor, face_library)

        if name:
            # Draw a rectangle and show identity
            x, y, w, h = det.left(), det.top(), det.width(), det.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} ({distance:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            log_event(LOG_FILE, name)
        else:
            # Mark unknown
            x, y, w, h = det.left(), det.top(), det.width(), det.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "Unknown",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
    return frame


if __name__ == "__main__":
    # Load the detector and predictor
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(
        "./data/shape_predictor_68_face_landmarks.dat"
    )
    face_recognizer = dlib.face_recognition_model_v1(
        "./data/dlib_face_recognition_resnet_model_v1.dat"
    )
    face_library = load_face_library(
        "./imgs", detector, shape_predictor, face_recognizer
    )
    # # Open the video capture device
    # cap = cv2.VideoCapture(0)
    # # Set the frame rate
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # # Set the frame size
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # capture one frame from the camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    # detect and draw faces on the frame
    new_frame = detect_and_draw_faces(
        frame, detector, shape_predictor, face_recognizer, face_library
    )
    # display the frame
    cv2.imshow("Frame", new_frame)
    cv2.waitKey(0)
