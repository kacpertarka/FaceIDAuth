from fastapi import APIRouter, Depends, UploadFile, HTTPException, status, File
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.database import get_db
from src.authenticate.schemas import Token, UserCreate
from src.authenticate.crud import TokenCRUD, UserCRUD, User, oauth2_scheme


# create a new API router with prefix for authentication users
auth_router = APIRouter(prefix='/auth')

# dependency definition for token and database session
token_dependencies = Depends(TokenCRUD)
db_dependencies = Depends(get_db)


# dependency for user database to acces CRUD operation on user table
def user_db_dependencies(db: Session = db_dependencies):
    return UserCRUD(db)


# function to retrieve current user based on the provided token
def get_current_user(token: str = Depends(oauth2_scheme), token_crud: TokenCRUD = token_dependencies, db: Session = db_dependencies):
    token = token_crud.verify_access_token(token)
    user = db.query(User).filter(User.email == token.email).first()
    return user



@auth_router.post('/register')
async def register(user_data: UserCreate,
             user_crud: UserCRUD = Depends(user_db_dependencies)):
    email = user_data.email
    if user_crud.get_user_by_email(email):
        raise HTTPException(status_code=status.HTTP_409_BAD_REQUEST,
                            detail='User with given email already exists')
    
    return user_crud.create_user(user_data)


@auth_router.post('/login/password', response_model=Token)  # login by email-password
async def login(user_data: OAuth2PasswordRequestForm = Depends(), 
          user_crud: UserCRUD = Depends(user_db_dependencies),
          token_crud: TokenCRUD = token_dependencies):
    email = user_data.username  # email
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='Invalid credentials')
    user = user_crud.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='Account with give email does not exist')
    if not user_crud.verify_password(user_data.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='Incorrect password')
    
    return token_crud.create_tokens(data={'email': email})


@auth_router.post('/faceid/register')
async def faceid_register(file: UploadFile = File(...),
                          user_crud: UserCRUD = Depends(user_db_dependencies),
                          user = Depends(get_current_user)):
    try:
        message = await user_crud.add_user_face_id(user, file)
    except Exception as er:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,
                            detail='Failed to open image from camera')
    finally:
        await file.close()

    return {'message': message}


@auth_router.post('/login/faceid', response_model=Token)  # login by faceID
async def face_login(file: UploadFile,
               user_crud: UserCRUD = Depends(user_db_dependencies),
               token_crud: TokenCRUD = token_dependencies):
    try:
        user, msg = await user_crud.get_user_by_face_id(file)
        if msg and not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail=f'Cannot login by face ID. {msg}')
    except HTTPException as http_err:
        raise http_err
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,
                            detail=f'Failed to open image from camera. {err}')
    finally:
        await file.close()

    return token_crud.create_tokens(data={'email': user.email})
