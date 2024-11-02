from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class User(Base):
    __tablename__ = 'users'

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(50))
    last_name: Mapped[str] = mapped_column(String(50))

    email: Mapped[str] = mapped_column(String(128))
    password: Mapped[str] = mapped_column(String(128))

    user_face: Mapped['FaceUserID'] = relationship(back_populates='user')

    def __repr__(self) -> str:
        return f'{self.first_name} {self.last_name}'
    

class FaceUserID(Base):
    __tablename__ = 'user_face'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    face_vector: Mapped[str] = mapped_column(String(200))  # TODO: check how longh this vector should be

    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'))
    user: Mapped['User'] = relationship(back_populates='user_face', single_parent=True)

