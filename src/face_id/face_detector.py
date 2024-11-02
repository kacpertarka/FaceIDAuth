import json
import time
import math
from typing import Optional

import cv2 as cv
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from src.face_id.face_analizator import FaceAnalizator


class FaceDetector:
    def __init__(self):
        # initialize FaceAnalizator for face vector analysis
        self.face_analizator = FaceAnalizator()

        # import necessary MediaPipe components 
        self.mp_face_mesh = solutions.face_mesh
        self.drawing = solutions.drawing_utils
        self.drawing_styles = solutions.drawing_styles

        # configurate MediaPipe FaceMesh with specific parameter
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_face(self, frame):
        """
        Process a frame to detect face landmarks
        """
        result = self.face_mesh.process(frame)
        return result
    
    def count_face_vector(self) -> list[float]:
        """
        Call the face vector analysis function
        """
        return self.face_analizator.count_face_vector()

    def verify_face(self, camera_vector: list[float], face_db_data: list[list]) -> tuple[str, float]:
        """
        Verify the face with the camera vector using face database
        
        0 <= similiarity <= 1
        1 means perfect match, 0 means no match

        """
        # initialize list to storage user_id and similiarity ratio
        similiarity_ratio: list[tuple[str, float]] = []  # storage user_id and siiliarity ratio
        
        for face_data in face_db_data:
            # convert the storage face vector data from JSON format
            face_db_vector = json.loads(face_data.face_vector)  # TODO: change table name 
            
            # calculate similiarity the camera face vector and database face vector and storage result into list
            similiarity = self.face_analizator.count_sequence_similiarity(camera_vector, face_db_vector)
            similiarity_ratio.append((face_data.user_id, similiarity))

        # find the user with the highest similiarity
        return max(similiarity_ratio, key=lambda x: x[1])        
