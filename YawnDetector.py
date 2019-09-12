import cv2
import dlib
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
cascade_path = 'Haarcascades/haarcascade_frontalface_default.xml'
casecase = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    print("Too Many Faces")
    pass

class NoFaces(Exception):
    print("No Faces Detected")
    pass


# Return facial landmarks of img as np array
def get_landmarks(img, dlibOn):
    if (dlibOn):
        rects = detector(img, 1)
        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces
        x,y,w,h = rects[0]
        rect = dlib.rectangle(x, y, x+w, y+h)
        return np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])

# Annotate landmarks with integer IDs
def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos, fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale = 0.4, color = (0, 0, 255))
        cv2.circle(img, pos, 3, color = (0, 255, 255))
    return img


# Return the mean position of the top lip using facial landmarks
def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis = 0)
    return int(top_lip_mean[:,1])

# Return mean position of the bottom lip using facial landmarks
def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis = 0)
    return int(bottom_lip_mean[:,1])

# Return the distance between top and bottom lip
def lip_distance(img):
    landmarks = get_landmarks(img)
    if (landmarks == "error"):
        return img, 0
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_difference = abs(top_lip_center - bottom_lip_center)
    annotated_image = annotate_landmarks(image)
    return lip_difference, annotated_image

# Set up video capture from webcam
cap = cv2.VideoCapture(0)
yawns = 0
is_yawning = False

while True:
    ret, frame = cap.read()
    lip_distance, image_landmarks = mouth_open(frame)
    prev_yawn_status = is_yawning
    if lip_distance > 25:
        is_yawning = True
        cv2.putText(image_landmarks "Subject is Yawning", (50, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 2)
    else:
        is_yawning = False
    if is_yawning and prev_yawn_status = False:
        yawns += 1

    output_text = "Number of Yawns: " + str(yawns)
    cv2.putText(image_landmarks), output_text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Live Yawn Detection', image_landmarks)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
