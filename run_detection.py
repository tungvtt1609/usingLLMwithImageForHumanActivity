import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import queue
from tensorflow.keras.models import load_model
from datetime import datetime
import os

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
    
class ActionDetector:
    def __init__(self):
        try:
            self.pose_detector = PoseDetector()
            self.model = load_model('pose_model.pth') 
            
            self.input_shape = self.model.input_shape
             # Load your trained model here

