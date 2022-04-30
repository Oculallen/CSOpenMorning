import mediapipe as mp
import numpy as np
import pantilthat
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

MOVE_CONSTANT = 4
# PAN = pantilthat.get_pan()
# TILT = pantilthat.get_tilt()
PAN = 0
TILT = 0

#Classes

#Functions
def rad_to_deg(val):
    return val * (180/math.pi)

def get_camera_center(landmarks):
    coords = []
    for face in landmarks:
        coords.append([int(face.landmark[6].x * width), int(face.landmark[6].y * height)])
    np_coords = np.array(coords)
    totals = np.sum(np_coords, axis=0)
    #print(np_coords.shape[0])
    totals = [int(totals[0]/np_coords.shape[0]), int(totals[1]/np_coords.shape[0])]
    #print(totals)
    return totals

def calc_camera_moves(landmarks):
    center = get_camera_center(landmarks)
    shift_x = -(center[0] - img_x)
    shift_y = -(center[1] - img_y)
    #print(shift_x, shift_y)
    
    if shift_x < -10:
        angle_x = MOVE_CONSTANT
    elif shift_x > 10:
        angle_x = -MOVE_CONSTANT
    else:
        angle_x = 0

    if shift_y < -10:
        angle_y = -MOVE_CONSTANT
    elif shift_y > 10:
        angle_y = MOVE_CONSTANT
    else:
        angle_y = 0

    return (angle_x, angle_y), center

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

        angles, center = calc_camera_moves(results.multi_face_landmarks)

    return img_final, angles, center

def limiter(angle):
    if angle > 90:
        return 90
    elif angle < -90:
        return -90
    else:
        return angle

def get_pos():
    return pantilthat.get_pan(), pantilthat.get_tilt()

def update_gimbal(angle_x, angle_y):
    try:
        PAN, TILT = get_pos()
        new_x = limiter(PAN + angle_x)
        new_y = limiter(TILT + angle_y)

        pantilthat.pan(new_x)
        pantilthat.tilt(new_y)

        return True
    except BaseException as e:
        return e 

#Main
def main():
    with mp_face_mesh.FaceMesh(max_num_faces=20, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            success, image = cap.read()
            
            if success:
                try:
                    img, angles, center = scan_for_faces(image, face_mesh)
                    print((angles, center))
                    img = cv2.circle(img, center, 10, (255, 0, 0))
                    cv2.imshow("Testing testing", img)

                    # gimbal_status = update_gimbal(angle_x=angles[0], angle_y=angles[1])
                    # if gimbal_status!= True:
                    #     print(gimbal_status)
                    #     break
                except BaseException as e:
                    print(e)
                    break
            
            # Terminate the process
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

if __name__ == "__main__":
    main()