import cv2
import mediapipe as mp
import screen_brightness_control as sbc

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get the current brightness of the first display (assumes a single display setup)
current_brightness = sbc.get_brightness(display=0)[0]
previous_y = None

try:
    while True:
        # Capture frames from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        result = hands.process(rgb_frame)

        # Draw hand landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the y-coordinate of the wrist (landmark 0)
                wrist_y = hand_landmarks.landmark[0].y

                if previous_y is not None:
                    # Compare current y position with previous y position
                    delta_y = wrist_y - previous_y

                    # Adjust brightness based on hand movement
                    if delta_y < -0.01:  # Hand moved up
                        current_brightness = min(current_brightness + 5, 100)  # Increase brightness by 5%, cap at 100%
                    elif delta_y > 0.01:  # Hand moved down
                        current_brightness = max(current_brightness - 5, 0)  # Decrease brightness by 5%, cap at 0%

                    sbc.set_brightness(current_brightness)

                # Update previous y-coordinate
                previous_y = wrist_y

        # Display the frame
        cv2.imshow("Gesture Brightness Control", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
