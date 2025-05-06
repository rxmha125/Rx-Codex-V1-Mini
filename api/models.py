# api/models.py (Corrected with Imports)

from pydantic import BaseModel, Field, EmailStr
from typing import Optional # <<< Added this import

class UserBase(BaseModel):
    username: str = Field(..., min_length=3)
    email: Optional[EmailStr] = None # Now Optional and EmailStr are defined

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserInDB(UserBase):
    hashed_password: str

class UserPublic(UserBase):
     # Add fields safe to return publicly if needed
     pass

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None