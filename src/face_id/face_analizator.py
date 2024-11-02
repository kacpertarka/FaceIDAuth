import numpy as np

from src.face_id.face_points import FacePoints


class FaceAnalizator:
    def __init__(self):
        """
        Initialize FaceAnalizator with an empty face vector.
        The face vector will store proportions of various facial features normalized by face dimensions.
        """
        self.face_vector: list[float] = []

    def post_init(self, frame, detected_faces):
        """
        Store the frame shape and detected faces for further processing
        """
        self.shape = frame.shape 
        self.detected_faces = detected_faces

    def count_face_vector(self):
        """
        Calculate the face vector based on various facial measurements.
        This method populates the face_vector attribute with normalized feature sizes and areas.
        """
        self.face_vector = []

        # calculate widths and heights of various facial features
        left_eye_width = self.count_points_width(FacePoints.FACEMESH_LEFT_EYE_OUTLYING_POINTS)
        right_eye_width = self.count_points_width(FacePoints.FACEMESH_RIGHT_EYE_OUTLYING_POINTS)
        left_eye_height = self.count_points_width(FacePoints.FACEMESH_LEFT_EYE_HEIGHT_POINTS)
        right_eye_height = self.count_points_width(FacePoints.FACEMESH_RIGHT_EYE_HEIGHT_POINTS)
        left_eyebrow = self.count_points_width(FacePoints.FACEMESH_LEFT_EYEBROW_POINTS)
        right_eyebrow = self.count_points_width(FacePoints.FACEMESH_RIGHT_EYEBROW_POINTS)

        eyes_distance = self.count_points_width(FacePoints.FACEMESH_EYES_DISTANCE_POINTS)

        nose_height = self.count_points_width(FacePoints.FACEMESH_NOSE_HEIGHT_POINTS)

        lips_width = self.count_points_width(FacePoints.FACEMESH_LIPS_WIDTH_POINTS)
        nose_to_lips = self.count_points_width(FacePoints.FACEMESH_NOSE_TO_LIPS_POINT)

        left_eye_to_left_lips = self.count_points_width(FacePoints.FACEMESH_LEFT_EYE_TO_LEFT_EDGE_OF_LIPS)
        right_eye_to_right_lips = self.count_points_width(FacePoints.FACEMESH_RIGHT_EYE_TO_RIGHT_EDGE_OF_LIPS)

        # face dimensions
        face_width = self.count_points_width(FacePoints.FACEMESH_FACE_OUTLYING_POINTS)
        face_height = self.count_points_width(FacePoints.FACEMESH_FACE_HEIGHT_POINTS)

        # face features area - normalized by whole face area
        face_area = self._calculate_polygon_area(FacePoints.FACEMESH_FACE_POLYGON)
        mouth_area = self._calculate_polygon_area(FacePoints.FACEMESH_MOUTH_POLYGON)
        nose_area = self._calculate_polygon_area(FacePoints.FACEMESH_NOSE_POLYGON)
        forehead_area = self._calculate_polygon_area(FacePoints.FACEMESH_FOREHEAD_POLYGON)
        nose_lips_triangle_are = self._calculate_polygon_area(FacePoints.FACEMESH_NOSE_LIPS_TRIANGLE)
        forehead_triangle_are = self._calculate_polygon_area(FacePoints.FACEMESH_FOREHEAD_TRIANGLE)
        left_cheek_area = self._calculate_polygon_area(FacePoints.FACEMESH_LEFT_CHEEK_POLYGON)
        right_cheek_area = self._calculate_polygon_area(FacePoints.FACEMESH_RIGHT_CHEEK_POLYGON)
        eyes_area = self._calculate_polygon_area(FacePoints.FACEMESH_LEFT_EYE_POLYGON) + self._calculate_polygon_area(FacePoints.FACEMESH_RIGHT_EYE_POLYGON)
        chin_area = self._calculate_polygon_area(FacePoints.FACEMESH_CHIN_POLYGON)

        # calculate angles
        eyes_vector = FacePoints.FACEMESH_LEFT_EYE_POINT + FacePoints.FACEMESH_RIGHT_EYE_POINT
        mouth_vector = FacePoints.FACEMESH_LEFT_MOUTH + FacePoints.FACEMESH_RIGHT_MOUTH
        eyes_mouth_angle = self._calculate_angle_from_face_points(FacePoints.FACEMESH_LIPS_POINT, eyes_vector)
        mouth_forehead_angle = self._calculate_angle_from_face_points(FacePoints.FACEMESH_FOREHEAD_CENTER_POINT, mouth_vector)
        mouth_top_forehead_angle = self._calculate_angle_from_face_points(FacePoints.FACEMESH_FOREHEAD_TOP_POINT, mouth_vector)

        self.face_vector.extend([
            face_width / face_height, 
            # features size noralized by face width
            nose_height / face_width,
            left_eye_width / face_width,
            right_eye_width / face_width,
            lips_width / face_width,
            eyes_distance / face_width,
            left_eye_to_left_lips / face_width,
            right_eye_to_right_lips / face_width,
            left_eyebrow / face_width,
            right_eyebrow / face_width,
            # features size normalized by nose height
            lips_width / nose_height,
            nose_to_lips / nose_height, 

            left_eye_height / left_eye_width,
            right_eye_height / right_eye_width,
            
            # areas normalized by face area
            mouth_area / face_area,
            nose_area / face_area,
            forehead_area / face_area,
            nose_lips_triangle_are / face_area,
            forehead_triangle_are / face_area,
            (left_cheek_area + right_cheek_area) / face_area,
            (nose_area + left_cheek_area) / face_area,
            (nose_area + right_cheek_area) / face_area,
            (nose_lips_triangle_are + left_cheek_area) / face_area,
            (nose_lips_triangle_are + right_cheek_area) / face_area,
            (nose_lips_triangle_are + forehead_area) / face_area,

            # areas normalized by other features
            eyes_area / nose_area,
            nose_area / forehead_area,
            left_cheek_area / forehead_area,
            right_cheek_area / forehead_area,
            chin_area / forehead_area,


            # angles between points
            eyes_mouth_angle,
            mouth_forehead_angle,
            mouth_top_forehead_angle,
        ])
        self.face_vector = np.array(self.face_vector) * 100
        return self.face_vector.astype(int)

    def count_points_width(self, points: list[int]) -> float:
        """
        Calculate dsitance between two facial points. Useing frame shape.
        """
        height, width, _ = self.shape

        # TODO: only one face should be detected
        for face_landmarks in self.detected_faces.multi_face_landmarks:
            left_landmark = face_landmarks.landmark[points[0]]
            right_landmark = face_landmarks.landmark[points[1]]
            
            # convert landmark coordinates to pixel values
            left_point = np.array([left_landmark.x * width, left_landmark.y * height])
            right_point = np.array([right_landmark.x * width, right_landmark.y * height])
            return np.linalg.norm(left_point - right_point)
        
    def _calculate_polygon_area(self, face_points: list[int]) -> float:
        """
        Calculate area of any polygon by numpy methods
        points: [(1, 2), (2, 3)]
        """
        face_coords = self._count_coords_from_face_points(face_points)
        x, y = np.hsplit(np.array(face_coords), 2)

        # calculate area of the polygon using the shoelace formula
        return float(0.5 * np.abs(np.dot(x.T, np.roll(y, 1)) - np.dot(y.T, np.roll(x, 1))))

    def _calculate_angle_from_face_points(self, center_point: list[int], top_points: list[int]) -> float:
        """
        Calculate the angle formed by the center point and two top points.
        """
        center_landmark = self._count_coords_from_face_points(center_point)[0]
        left_landmark = self._count_coords_from_face_points([top_points[0]])[0]
        right_landmark = self._count_coords_from_face_points([top_points[1]])[0]


        # create vector
        AB = np.array([center_landmark[0] - left_landmark[0], center_landmark[1] - left_landmark[1]])
        BC = np.array([right_landmark[0] - left_landmark[0], right_landmark[1] - left_landmark[1]])

        # calcuate of the scalar product and norm of vectors
        dot_product = np.dot(AB, BC)
        norm_AB = np.linalg.norm(AB)
        norm_BC = np.linalg.norm(BC)

        # calculate of angles in radian
        angle_rad = np.arccos(dot_product / (norm_AB * norm_BC))
        
        # return angle in radians
        return angle_rad 
    
    def _count_coords_from_face_points(self, face_points: list[int]) -> list[tuple[int, int]]:
        """
        Get the pixel coordinates of specified facial landmarks.
        """
        height, width, _ = self.shape

        # TODO: only one face should be detected
        for face_landmarks in self.detected_faces.multi_face_landmarks:
            points: list = []
            for idx in face_points:
                x = int(face_landmarks.landmark[idx].x * width)
                y = int(face_landmarks.landmark[idx].y * height)
                points.append((x, y))
            return points
    
    def count_sequence_similiarity(self, current_vector: list[float], db_vector: list[float]) -> float:
        """
        Calculate similarity between two vectors using sequence similarity method. 
        Return similiarity rounded to 5 places
        """
        current_vector = np.array(current_vector)
        db_vector = np.array(db_vector)
        ratio = np.dot(current_vector, db_vector) / (np.linalg.norm(current_vector) * np.linalg.norm(db_vector))
        return round(ratio, 5)

