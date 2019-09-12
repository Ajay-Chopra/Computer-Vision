import cv2
import dlib
import numpy as np
from time import sleep
import sys

# Import trained model
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up images
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS +
                + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from second image to be overlayed onto the first
# The convex hull of each will be overlaid
OVERLAY_POINTS = [LEFT_EYE_POINTS, RIGHT_EYE_POINTS, LEFT_BROW_POINTS,
                 RIGHT_BROW_POINTS, NOSE_POINTS, MOUTH_POINTS]

# Amount of blur to use during color correction
COLOR_CORRECT_BLUR_FRAC = 0.6
cascade_path = 'Haarcascades/haarcascade_frontalface_default.xml'
casecase = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Given an image, return facial landmarks as np array
def get_landmarks(img, dlibOn):
    if (dlibOn):
        rects = detector(img, 1)
        if len(rects) > 1:
            print("Error in get_landmarks: More than one face detected")
            return
        if len(rects) == 0:
            print("Error in get_landmarks: No faces detected")
            return
        x,y,w,h = rects[0]
        rect = dlib.rectangle(x, y, x+w, y+h)
        return np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])

# Annotate facial landmarks with individual integer indentifier
def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos, fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale = 0.4, color = (0, 0, 255))
        cv2.circle(img, pos, 3, color = (0, 255, 255))
    return img


def draw_convex_hull(img, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color = color)

# Return mask hilighting region containing eyes, nose and mouth
def get_face_mask(img, landmarks):
    img = np.zeros(img.shape[:2], dtype = np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(img, landmarks[group], color = 1)

    img = np.array([img, img, img]).transpose((1, 2, 0))
    img = (cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    img = cv2.GuassianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return img

# Use procrustes analysis to align the two faces
def transformation_from_points(points_1, points_2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis = 0)
    c2 = np.mean(points2, axis = 0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                                 np.matrix([0., 0., 1.])])


def read_img_and_landmarks(fname):
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx = 0.35, fy = 0.35, interpolation = cv2.INTER_LINEAR)
    img = cv2.resize(img, (img.shape[1] * SCALE_FACTOR, img.shape[0] * SCALE_FACTOR))
    s = get_landmarks(img, dlibOn)
    return img, s

# Use warpAffine to overlay one image onto the other
def warp_img(img, M, dshape):
    output_img = np.zeros(dshape, dtype = img.dtype)
    cv2.warpAffine(img, M[:2], (dshape[1], dshape[0]), dst = output_img,
                    borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP)
    return output_img

# Correct the colors at the edges of the facial overlay by implementing GuassianBlur
def correct_colors(img1, img2, landmarks1):
    blur_amount = COLOR_CORRECT_BLUR_FRAC * np.linalg.norm(
                                            np.mean(landmarks1[LEFT_EYE_POINTS], axis = 0) -
                                            np.mean(landmarks1[RIGHT_EYE_POINTS], axis = 0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    img2_blur += (128 * (img2_blur <= 1.0)).astype(im2_blur.dtype)

    return (img2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64))

# Final faceswap with which maps the face of img2 onto that of img1
def face_swap(img, name):
    s = get_landmarks(img, True)
    img1, landmarks1 = img, s
    img2, landmarks2 = read_img_and_landmarks(name)

    M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
    mask = get_face_mask(img2, landmarks2)
    warped_mask = warp_img(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(img1, landmarks1), warped_mask], axis = 0)

    warped_img2 = warp_img(img2, M, img1.shape)
    warped_corrected_img2 = correct_colors(img1, warped_img2, landmarks1)
    output_img = img1 * (1.0 - combined_mask) + warped_corrected_img2 * combined_mask

    cv2.imwrite('output.jpg', output_img)
    img = cv2.imread('output.jpg')
    frame = cv2.resize(img, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_LINEAR)

    return img

cap = cv2.VideoCapture(0)
filter_image = "./images/testImg1.jpg"
dlibOn = False

while True:
    ret, frame = cap.read()

    # Reduce the image size by 75% to improve frame rate
    frame = cv2.resize(frame, None, fx = 0.75, fy = 0.75, interpolation = CV2.INTER_LINEAR)
    # Flip image to make the swap more mirror like
    frame = cv2.flip(frame, 1)
    cv2.imshow('Live Face Swapper', face_swap(frame, filter_image))

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
