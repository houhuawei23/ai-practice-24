# import gradio as gr
from face_input import face_input, test_face_input
import cv2

from tkinter import Tk, Label, Button, filedialog, Canvas
from PIL import Image, ImageTk
import dlib

from face import detect_and_draw_faces, load_face_library
from log import log_event

LOG_FILE = "./access_logs.txt"
# Global variable for capturing video
cap = None

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1(
    "./data/dlib_face_recognition_resnet_model_v1.dat"
)
face_library = load_face_library("./imgs", detector, shape_predictor, face_recognizer)


def start_camera_ui(canvas: Canvas, video_label: Label):
    global cap

    if cap is not None and cap.isOpened():
        return

    cap = cv2.VideoCapture(0)  # Open the camera

    def update_frame():
        # print("update_frame")
        global cap
        if cap is None or not cap.isOpened():
            img_tk = None
        else:
            ret, frame = cap.read()
            if ret:
                # Process the frame for face recognition
                processed_frame = detect_and_draw_faces(
                    frame, detector, shape_predictor, face_recognizer, face_library
                )
                # processed_frame = frame

                # Convert OpenCV image (BGR) to PIL image (RGB)
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new frame
        video_label.img_tk = img_tk
        video_label.configure(image=img_tk)

        # Schedule the next frame update
        canvas.after(10, update_frame)

    update_frame()  # Start updating frames


# Simple GUI with Tkinter
def browse_image(processed_label: Label):
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (640, 480))
        # Process the frame for face recognition
        processed_frame = detect_and_draw_faces(
            img, detector, shape_predictor, face_recognizer, face_library
        )
        # Convert OpenCV image (BGR) to PIL image (RGB)
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        processed_label.img_tk = img_tk
        processed_label.configure(image=img_tk)


def stop_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None


from human_check import check_human_face


def create_ui():
    root = Tk()
    root.title("Face Recognition Access Control")
    root.geometry("800x600")

    Label(root, text="Face Recognition Access Control System", font=("Arial", 16)).pack(
        pady=10
    )

    # Create a canvas for video feed
    canvas = Canvas(root, width=640, height=480)
    canvas.pack(pady=10)

    # Add a label to display the video frames
    video_label = Label(canvas)
    video_label.pack()

    Button(
        root,
        text="Start Camera",
        command=lambda: start_camera_ui(canvas, video_label),
        width=20,
    ).pack(pady=5)

    # stop button
    Button(
        root,
        text="Stop Camera",
        command=lambda: (stop_camera(), video_label.configure(image=None)),
        width=20,
    ).pack(pady=5)

    Button(
        root,
        text="Capture Faces",
        command=lambda: test_face_input(cap, detector),
        width=20,
    ).pack(pady=5)



    # Add a label to display the processed frames
    image_label = Label(root, text="No Image Selected", bg="gray")
    image_label.pack(pady=10)

    Button(
        root,
        text="Process Image",
        command=lambda: browse_image(image_label),
        width=20,
    ).pack(pady=5)


    # human face recognition
    Button(
        root,
        text="Human Face Recognition",
        command=lambda: check_human_face(cap, canvas, video_label),
        width=20,
    ).pack(pady=5)
    # root.protocol(
    #     "WM_DELETE_WINDOW", lambda: (stop_camera(), root.destroy())
    # )  # Handle window close

    root.mainloop()


if __name__ == "__main__":
    # demo.launch()
    # face_input()
    # test_face_input()
    create_ui()
