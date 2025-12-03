import cv2
import numpy as np
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Draw a cube on the frame
def draw_cube(frame, center_x, center_y, scale):
    cube_points = np.float32([
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
        [1, -1, 1]
    ]) * scale

    cube_points[:, 0] += center_x
    cube_points[:, 1] += center_y

    cube_points = cube_points.astype(int)

    # Cube edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        pt1 = tuple(cube_points[edge[0]][0:2])
        pt2 = tuple(cube_points[edge[1]][0:2])
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

cap = cv2.VideoCapture(0)

cube_x, cube_y = 300, 200  # Start position
cube_scale = 80            # Cube size

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark

            # Index finger tip (point 8)
            x = int(lm[8].x * w)
            y = int(lm[8].y * h)

            # Thumb tip (point 4) for zoom
            thumb_x = int(lm[4].x * w)
            thumb_y = int(lm[4].y * h)

            # Gesture-based 3D control
            cube_x = int(x)
            cube_y = int(y)

            distance = math.hypot(x - thumb_x, y - thumb_y)
            cube_scale = int(40 + distance)  # Zoom cube

            # Draw tracking points
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)

        draw_cube(frame, cube_x, cube_y, cube_scale)
        cv2.putText(frame, "Move finger to move cube | Pinch to zoom",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("3D Gesture Mover", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
