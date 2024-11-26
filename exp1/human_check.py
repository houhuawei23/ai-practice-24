import cv2
import dlib
from PIL import Image, ImageTk
from tkinter import Canvas, Label

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


def check_human_face(cap, canvas: Canvas, video_label: Label):
    """
    Continuously checks if the face in the video feed is dynamic and performs a task (e.g., smiling).
    """

    if not cap or not cap.isOpened():
        print("Camera is not running.")
        return

    def process_video():
        ret, frame = cap.read()
        if not ret:
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_frame)

        for face in faces:
            # Draw the detected face
            cv2.rectangle(
                frame,
                (face.left(), face.top()),
                (face.right(), face.bottom()),
                (0, 255, 0),
                2,
            )

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

        return frame

    process_video()

    while True:
        frame = process_video()
        cv2.imshow("Human Check", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
