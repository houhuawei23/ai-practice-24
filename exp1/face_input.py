import cv2
import dlib
import sys
import os


def test_face_input(camera, detector):
    """
    test face input function
    """
    name = input("Please Enter your name: ")
    imgs_path = "./imgs/"
    face_input(camera, detector, name, imgs_path)


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def face_input(camera: cv2.VideoCapture, detector, name="unknown", base_path="./imgs"):
    """
    capture face input from camera and save to local directory
    """
    # detector = dlib.get_frontal_face_detector()

    new_path = os.path.join(base_path, name)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    # camera = cv2.VideoCapture(0)

    if camera is None or not camera.isOpened():
        print("Error: Could not open camera")
        return

    cnt = 0
    while cnt < 10:
        print("in process of enter face %s" % cnt)
        readsuccess, img = camera.read()

        if not readsuccess:
            print("Error: Could not read image from camera")
            return

        # faces: Dict[int, dlib.rectangle]
        faces = detector(img, 1)

        # sort faces by area
        faces = sorted(faces, key=lambda x: x.area(), reverse=True)

        for i, d in enumerate(faces):
            x, y, w, h = d.left(), d.top(), d.width(), d.height()

            if i == 0:
                color = RED
            else:
                color = GREEN
                # cv2.putText(img, "Press 'S' to save face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(
                img,
                "Press 'S' to save face",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("img", img)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        if key & 0xFF == ord("s"):
            print("Saving image...")
            if len(faces) > 1:
                print("More than one face detected!")
            elif len(faces) == 0:
                print("No face detected!")
            else:
                cnt += 1
                face = cv2.resize(img, (128, 128))
                cv2.imwrite(os.path.join(new_path, name + str(cnt) + ".jpg"), face)

    print("Done!")
    # camera.release()
    cv2.destroyAllWindows()
