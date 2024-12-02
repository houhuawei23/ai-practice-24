# importing the module
import cv2


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, " ", y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow("image", img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, " ", y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(
            img, str(b) + "," + str(g) + "," + str(r), (x, y), font, 1, (255, 255, 0), 2
        )
        cv2.imshow("image", img)

import os

# driver function
if __name__ == "__main__":
    # Paths to data
    template_path = "./LicensePlateDataTest/Character_templates"
    plate_path = "./LicensePlateDataTest/License_plate"
    plate_path = os.path.join(plate_path, "ahy1N6L77_blue_False.jpg")

    # reading the image
    img = cv2.imread(plate_path, 1)

    # displaying the image
    cv2.imshow("image", img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback("image", click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
