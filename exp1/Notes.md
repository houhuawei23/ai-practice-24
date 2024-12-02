# opencv-python

haarcascade_frontalface_default.xml: Trained XML classifiers describes some features of some object we want to detect a cascade function is trained from a lot of positive(faces) and negative(non-faces) images.

### Image

Read an image from file (using `cv::imread`)
Display an image in an OpenCV window (using `cv::imshow`)
Write an image to a file (using `cv::imwrite`)

`IMREAD_COLOR` loads the image in the BGR 8-bit format. This is the default that is used here.
`IMREAD_UNCHANGED` loads the image as is (including the alpha channel if present)
`IMREAD_GRAYSCALE` loads the image as an intensity one

`core` section, as here are defined the basic building blocks of the library
`imgcodecs` module, which provides functions for reading and writing
`highgui` module, as this contains the functions to show an image in a window

### Video

VideoCapture

VideoWriter

### Drawing

- Learn to draw different geometric shapes with OpenCV
- You will learn these functions :
  - cv.line(), cv.circle() , cv.rectangle(), cv.ellipse(), cv.putText() etc.

img, color, thikness, linType

### Mouse

Learn to handle mouse events in OpenCV
You will learn these functions : `cv.setMouseCallback()`

['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON', 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK', 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL', 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

### Trackbar

For `cv.createTrackbar()` function, first argument is the trackbar name, second one is the window name to which it is attached, third argument is the default value, fourth one is the maximum value and fifth one is the callback function which is executed every time trackbar value changes. The callback function always has a default argument which is the trackbar position. In our case, function does nothing, so we simply pass.

Another important application of trackbar is to use it as a button or switch. OpenCV, by default, doesn't have button functionality. So you can use trackbar to get such functionality. In our application, we have created one switch in which application works only if switch is ON, otherwise screen is always black.

## Operations on Images

- Basic Operations on Images

  - Learn to read and edit pixel values, working with image ROI and other basic operations.

- Arithmetic Operations on Images

  - Perform arithmetic operations on images

- Performance Measurement and Improvement Techniques

  - Getting a solution is important. But getting it in the fastest way is more important. Learn to check the speed of your code, optimize the code etc.

### Image Processing

### Feature Detection and Description

### Video Analysis

# Dlib

local build and install:

pkg-config --cflags --libs dlib-1
-I/usr/local/include -L/usr/local/lib -ldlib /usr/lib/x86_64-linux-gnu/libsqlite3.so

apt install:

/usr/include/dlib

`get_frontal_face_detector()`

This function returns an `object_detector` that is configured to find human faces that are looking more or less towards the camera. It is created using the `scan_fhog_pyramid` object.

`shape_predictor`

One Millisecond Face Alignment with an Ensemble of Regression Trees, CVPR 2014

这篇论文解决了单张图像的人脸对齐问题。我们展示了如何使用回归树集成直接从像素强度的稀疏子集估计人脸的关键点位置，从而实现超实时的高质量预测。我们提出了一种基于梯度提升的通用框架，用于学习回归树集成，优化平方误差损失和自然处理缺失或部分标注的数据。我们展示了如何利用适当的先验信息，利用图像数据的结构来帮助高效的特征选择。我们还研究了不同的正则化策略及其在防止过拟合中的重要性。此外，我们分析了训练数据量对预测精度的影响，并探讨了使用合成数据进行数据增强的效果。

python bindings:

`class dlib.image_window`

This is a GUI window capable of showing images on the screen.

add_overlay(rectangles, color=rgb_pixel(255,0,0)) -> None
add_overlay(rectangle, color=rgb_pixel(255,0,0)) -> None
add_overlay(full_object_detection, color=rgb_pixel(255,0,0)) -> None

clear_overlay()

get_next_double_click(self: dlib.image_window) -> object

get_next_keypress()

is_closed() -> bool

set_image(img: numpy.ndarray[(rows, cols), int]) -> None

set_title(title: str) -> None

wait_until_closed() -> None

wait_for_keypress(key: str) -> int
Blocks until the user presses the given key or closes the window.

`class dlib.face_recognition_model_v1`

This object maps human faces into 128D vectors where pictures of the same person are mapped near to each other and pictures of different people are mapped far apart. The constructor loads the face recognition model from a file.

```python
def compute_face_descriptor(
  img: numpy.ndarray[(rows, cols, 3), uint8],
  face: full_object_detection,
  num_jitters: int=0,
  padding: float=0.25),
  -> dlib.vector
```

Takes an image and a `full_object_detection` that references a face in that image and converts it into a `128D face descriptor`. If `num_jitters>1` then each face will be randomly jittered slightly `num_jitters` times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.

dlib.vector
This object is an array of vector objects.

# shape_predictor_68_face_landmarks

[facial-point-annotations ](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
