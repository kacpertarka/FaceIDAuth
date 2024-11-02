from typing import Optional
from datetime import timedelta, datetime, timezone
import uuid
import json

import numpy as np
import cv2
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext
from fastapi import HTTPException, status, UploadFile
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from src.authenticate.schemas import UserCreate, UserResponse, TokenData, Token, FaceUserCreate
from src.authenticate.models import User, FaceUserID
from src.database import get_db
from src.face_id.face_detector import FaceDetector
from utils import load_env, load_env_as_number


# load configuration keys
SECRET_KEY = load_env('SECRET_KEY')
ALGORITHM = load_env('ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = load_env_as_number('ACCESS_TOKEN_EXPIRE_MINUTES')
REFRESH_TOKEN_EXPIRE_MINUTES = load_env_as_number('REFRESH_TOKEN_EXPIRE_MINUTES')

# set up OAuth2 password bearer token
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl='/auth/login/password'
)


class UserCRUD:
    """
    Class for managing CRUD operations for users.
    """
    def __init__(self, db: Session):
        self.db = db

        self.face_detector = FaceDetector()
        self.pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

    def create_user(self, user_schema: UserCreate) -> UserResponse:
        """
        create new user
        """
        raw_password = user_schema.password
        # hash raw password
        hashed_password = self._hash_password(raw_password)
        # unique user id
        id = str(uuid.uuid4())
        new_user = User(
            id=id,
            first_name=user_schema.first_name,
            last_name=user_schema.last_name,
            email=user_schema.email,
            password=hashed_password
        )
        try:
            self.db.add(new_user)
            self.db.commit()
            self.db.refresh(new_user)
        except IntegrityError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Error during creating new user. Try again later.')

        # return user response schema without password
        return UserResponse(
            id=new_user.id,
            first_name=new_user.first_name,
            last_name=new_user.last_name
        )
    
    def create_face_id(self, user_face: FaceUserCreate, db: Session):
        """
        create a face ID record for a user in the database.
        """
        face_data = user_face.model_dump()
        face_data['face_vector'] = json.dumps(face_data['face_vector'])
        # create new FaceUserID object
        new_face_user = FaceUserID(**face_data)
        self.db.add(new_face_user)
        self.db.commit()
        self.db.refresh(new_face_user)

        return new_face_user
    
    async def get_user_by_face_id(self, file: UploadFile) -> tuple[Optional[User], Optional[str]]:
        """
        Retrieves a user based on face ID data from the uploaded file.
        """
        # convert uploaded file to an image and detect face
        frame = await self._convert_bytes_to_image(file)
        result = self.face_detector.detect_face(frame)

        if result.multi_face_landmarks:
            # if face was detected
            # initialize face analyzer
            self.face_detector.face_analizator.post_init(frame, result)
            # calculate face vector
            face_vector = self.face_detector.count_face_vector()
            # get storaged face vectors from database
            db_results = self._get_user_face_ids()
            user_id, ratio = self.face_detector.verify_face(face_vector, db_results)
            if ratio < 0.99: # checking if user is recognized
                return None, 'User not recognised'
            
            # return found user
            return self.get_user_by_id(user_id), None
        # user was not found
        return None, 'Face not detected'
        
    async def add_user_face_id(self, user: User, file: UploadFile) -> str:
        """
        Adds face ID data to an existing user.
        """
        if user.user_face:
            return f'Facial recognition is already set up for this user. No additional registration is required.'
        
        # convert uploaded file to an image and detect face
        frame = await self._convert_bytes_to_image(file)
        result = self.face_detector.detect_face(frame)
        if result.multi_face_landmarks:
            self.face_detector.face_analizator.post_init(frame, result)
            face_vector = self.face_detector.count_face_vector()
            user_id = user.id
            face_user = self.create_face_id(FaceUserCreate(user_id=user_id, face_vector=face_vector), self.db)


            # update user with face ID record
            user.user_face = face_user
            self.db.commit()
            self.db.refresh(user)
            return 'Facial recognition login has been successfully added. You can now log in using facial recognition.'

        return 'Face has not detected on '

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Retrieves a user based on their email address."""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_user_by_id(self, user_id: str) -> Optional[User]: 
        """Retrieves a user based on their identifier."""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def verify_password(self, raw_password: str, db_password: str) -> bool:
        """Verifies a user's password."""
        return self.pwd_context.verify(raw_password, db_password)
    
    def _get_user_face_ids(self) -> list:
        """
        Retrieves all stored user face IDs from the FaceUserID table.
        Only user_id and face_vector columns
        """
        return self.db.query(FaceUserID).with_entities(FaceUserID.user_id, FaceUserID.face_vector).all()
    
    def _hash_password(self, raw_password: str) -> str:
        """Hashes the raw password."""
        return self.pwd_context.hash(raw_password)
    
    async def _convert_bytes_to_image(self, raw_data: bytes) -> np.array:
        """Converts raw byte data to an image."""
        # read raw data
        image_bytes = await raw_data.read()
        # create numpy array 
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        # decode the image
        image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class TokenCRUD:
    """Class for managing token operations."""
    def __init__(self):
        self.oauth2_scheme = oauth2_scheme

    def create_tokens(self, data: dict) -> Token:
        """Creates new tokens (access and refresh)."""
        return Token(
            access_token=self._create_access_token(data), 
            refresh_token=self._create_refresh_token(data),
            token_type='bearer'
        )

    def _create_access_token(self, data: dict, expire_token_time: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=expire_token_time)
        to_encode['expire'] = expire.strftime('%Y-%m-%d %H:%M:%S')

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, ALGORITHM)

        return encoded_jwt
    
    def _create_refresh_token(self, data: dict) -> str:
        return self._create_access_token(data, expire_token_time=REFRESH_TOKEN_EXPIRE_MINUTES)
    
    def verify_access_token(self, token: str): 
        credentional_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                               detail='Could not validate user token')
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)
            email = payload.get('email')

            if email is None:
                raise credentional_exception
            token = TokenData(email=email)

        except JWTError:
            raise credentional_exception
        
        return token
    
