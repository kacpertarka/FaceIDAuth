from typing import Optional

from pydantic import BaseModel, Field, EmailStr


class UserBase(BaseModel):
    first_name: str = Field(max_length=50)
    last_name: str = Field(max_length=50)


class UserResponse(UserBase):
    id: str


class UserLogin(BaseModel):
    email: str = Field(max_length=128)
    password: str = Field(max_length=128)



class UserCreate(UserBase):
    email: EmailStr
    password: str = Field(max_length=128)


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

    class Config:
        from_attributes=True


class TokenData(BaseModel):
    email: str
    

class FaceUserCreate(BaseModel):
    face_vector: list[float]
    user_id: str
