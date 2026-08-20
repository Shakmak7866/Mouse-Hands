import cv2 as cv
import time
import mediapipe as mp
import math
import pyautogui


def print_landmarks(mp_hands):
    for landmark in mp_hands.HandLandmark:
        print(landmark, landmark.value)


def calculate_FPS(prev_time, curr_time):
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    return prev_time, curr_time

def hand_draw(image, results, mp_drawing, mp_hands):
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

def check_distance(finger_a, finger_b):
    x = (finger_a.x - finger_b.x)
    y = (finger_a.y - finger_b.y)
    return math.sqrt( (x**2) + (y**2) )
    

def left_click(index, thumb):
    click_dist = check_distance(index, thumb)
    if click_dist < 0.05:
        pyautogui.mouseDown(button='left')
        time.sleep(1)
    else:
        pyautogui.mouseUp(button='left')
        time.sleep(1)

def right_click(middle, thumb):
    click_dist = check_distance(middle, thumb)
    if click_dist < 0.05:
        pyautogui.mouseDown(button='right')
        time.sleep(1)
    else:
        pyautogui.mouseUp(button='right')
        time.sleep(1)

def scroll_click(index, middle):
    click_dist = check_distance(index, middle)
    if click_dist < 0.05:
        pyautogui.mouseDown(button='middle')
        time.sleep(1)
    else:
        pyautogui.mouseUp(button='middle')
        time.sleep(1)


def hand_detection(
    results, 
    mp_drawing, 
    mp_hands, 
    hands, 
    screen_width, 
    screen_height, 
    prev_x,
    prev_y,
):
    hand = results.multi_hand_landmarks[0]

    # Index + Thumb = Left Click
    # Middle + Thumb = Right Click
    # Index + Middle = Scroll Button

    index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]

    # Mouse Following Code
    target_x = index.x * screen_width
    target_y = index.y * screen_height
    next_x = prev_x + 0.2 * (target.x - prev_x) # For easing into
    next_y = prev_y + 0.2 * (target.y - prev_y) # For easing into
    pyautogui.moveTo(next_x, next_y)
    prev_x = next_x
    prev_y = next_y

    left_click(index, thumb)
    right_click(middle, thumb)
    scroll_click(index, middle)



def main():
    # Set up utilites first

    # Pyautogui setup
    pyautogui.PAUSE = 0.1 # Makes the mouse move cleaner
    screen_width, screen_height = pyautogui.size()
    prev_x, prev_y = pyautogui.position()

    # Time setup
    prev_time = 0
    curr_time = 0

    # Camera Setup
    capture = cv.VideoCapture(0)

    # MediaPipe Setup
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands = 1, # We only want one hand at a time to move the mouse
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print_landmarks(mp_hands)

    # Loop for capturing hand detection
    while capture.isOpened():

        # Capture frame by frame and resize it
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (640, 480))

        # Improves image and performance
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Main Logic
        if results.multi_hand_landmarks:
            # Draw points on hand
            hand_draw(image, results, mp_drawing, mp_hands)

            # Move mouse to where hand is
            hand_detection(
                results, 
                mp_drawing, 
                mp_hands, 
                hands, 
                screen_width, 
                screen_height, 
                prev_x,
                prev_y,
            )

        # Show FPS
        prev_time, curr_time = calculate_FPS(prev_time, curr_time)

        # Show Camera and drawings
        cv.imshow("Mouse Hands", image)

        # Press q to quit
        if cv.waitKey(20) & 0xFF==ord('q'):
            break

    hands.close()
    capture.release()
    cv.destoryAllWindows()



if __name__ == "__main__":
    main()
