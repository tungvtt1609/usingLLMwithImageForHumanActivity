import cv2
import numpy as np
import os
import mediapipe as mp
import time

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img
    
    def get_landmarks(self, img):
        landmarks = []
        if self.results.pose_landmarks:
            for lm in self.results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    
def create_folder(actions):
    if not os.path.exists('pose_data'):
        os.makedirs('pose_data')
    for action in actions:
        if not os.path.exists(os.path.join('pose_data',action)):
            os.makedirs(os.path.join('pose_data',action))

def collect_data():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = PoseDetector()
    actions = ['standing', 'sitting', 'walking', 'running']
    create_folder(actions)

    sequency_length = 5
    sequency_per_action = 10   

    for action in actions:
        print(f"Collecting data for action: {action}")
        input("Press Enter to start...")
        for seq in range(sequency_per_action):
            print(f"Collecting sequence {seq+1}/{sequency_per_action} for action: {action}")
            for i in range(3, 0, -1):
                print(f"Starting in {i} seconds...")
                time.sleep(1)
            
            frame_data = []

            while len(frame_data) < sequency_length:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    break

                frame = detector.find_pose(frame)
                landmarks = detector.get_landmarks(frame)
                
                if landmarks:
                    frame_data.append(frame)  # Append the frame with landmarks drawn
                    cv2.putText(frame, f"Collecting {action}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Pose Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if len(frame_data) == sequency_length:
                for idx, frame in enumerate(frame_data):
                    frame_path = os.path.join('pose_data', action, f"{action}_{seq}_{idx}.jpg")
                    cv2.imwrite(frame_path, frame)
                print(f"Saved {action} sequence {seq+1}/{sequency_per_action} as images")
            
            time.sleep(1)
    cap.release
    cap.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
    print("Data collection completed.")
