import mediapipe as mp
import numpy as np
import cv2
import math

#INITIAL SETUP (DO NOT TOUCH)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

success, img_test = cap.read()
height, width, channels = img_test.shape
img_x = width // 2
img_y = height // 2
BASE_DIST = 20
text_mark = 69

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50) #Default
# fontScale
fontScale = 3
# Blue color in BGR
color = (0, 0, 255)
# Line thickness of 2 px
thickness = 2

#Functions
def limiter(val, limits):
    if val > limits[1]:
        val = limits[1]
    elif val < limits[0]:
        val = limits[0]
    return val

def calc_distance(landmarks):
    p1 = landmarks[13]
    p2 = landmarks[14]
    dist = int(math.sqrt(math.pow((p1.x - p2.x)*width, 2) + math.pow((p1.y - p2.y)*height, 2)))
    return dist

def calc_smile_percent(distance):
    return int(limiter(((distance/BASE_DIST) * 100), (0, 100)))

def get_text_center(landmarks):
    return int(landmarks[text_mark].x * width), int(landmarks[text_mark].y * height)

def get_colour(perc):
    return (0, int(255*(perc/100)), int(255*(1-(perc/100))))

def scan_for_faces(img, face_mesh):
    img_col = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_col.flags.writeable = False
    results = face_mesh.process(img_col)
    img_col.flags.writeable = True
    img_final = cv2.cvtColor(img_col, cv2.COLOR_RGB2BGR)
    angles = None
    center = None

    if results.multi_face_landmarks:     
        # for face_landmarks in results.multi_face_landmarks:
            # mp_drawing.draw_landmarks(
            # image=img_final,
            # landmark_list=face_landmarks,
            # connections=mp_face_mesh.FACEMESH_TESSELATION,
            # landmark_drawing_spec=None,
            # connection_drawing_spec=mp_drawing_styles
            # .get_default_face_mesh_tesselation_style())
        for landmarks in results.multi_face_landmarks:
            smile_dist = calc_distance(landmarks.landmark)
            smile_perc = calc_smile_percent(smile_dist)
            text_x, text_y = get_text_center(landmarks.landmark)
            org = (text_x, text_y)
            color = get_colour(smile_perc)
            img_final = cv2.putText(img_final, f'{smile_perc}%', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            print(smile_perc)

    return img_final

#Main
def main():
    with mp_face_mesh.FaceMesh(max_num_faces=20, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            success, image = cap.read()
            
            if success:
                #try:
                    img = scan_for_faces(image, face_mesh)
                    
                    cv2.imshow("Testing testing", img)
                    
                # except BaseException as e:
                #     print(e)
                #     break
            
            # Terminate the process
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

if __name__ == "__main__":
    main()