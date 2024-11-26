import cv2
import dlib
import sys
import os
def feature_extraction(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # fhog_object_detector object
    detector = dlib.get_frontal_face_detector()
    # shape_predictor object
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # This is a GUI window capable of showing images on the screen.
    # image_window object
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)

    # list of rectangle: List[dlib.rectangle]
    # class dlib.rectangles
    faces = detector(img, 1)

    print("Number of faces detected: {}".format(len(faces)))

    for k, d in enumerate(faces):
        print(
            "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()
            )
        )
        # class dlib.full_object_detection
        shape: dlib.full_object_detection = predictor(img, d)
        # shape.parts(): List[dlib.point] / class dlib.points
        parts = shape.parts()
        print(shape.parts())
        win.add_overlay(shape)

    win.add_overlay(faces)
    # Asks the user to hit enter to continue and pauses until they do so.
    dlib.hit_enter_to_continue()


# python feature_extraction.py ./imgs/hhw.jpg


"""
python binds:

point object:
    .x, .y

full_object_detection object: (shape)
    rectangle rect, vector<point> parts
    shape.num_parts, shape.parts(), shape.part(i), shape.rect

rectangle object:
    .left(), .top(), .right(), .bottom()
    .area(), 
    .tl_corner(), .tr_corner(), .bl_corner(), .br_corner(): return point object

frontal_face_detector:


"""

if __name__ == "__main__":
    # image_path = sys.argv[1]
    image_path = "./imgs/hhw.jpg"
    # print current directory
    print(os.getcwd())
    feature_extraction(image_path)
