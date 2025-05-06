# api/auth_utils.py (Corrected - Instructions Removed)

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from pydantic import ValidationError
import os
from dotenv import load_dotenv
from typing import Optional

# Import the Pydantic model if needed within this file (e.g., for type hints)
try:
    # Assumes models.py is in the same directory
    from .models import TokenData
except ImportError:
    # Fallback if run directly or structure differs
    from models import TokenData


load_dotenv() # Load variables from .env file

# --- Configuration ---
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

if not SECRET_KEY:
    raise EnvironmentError("JWT_SECRET_KEY not found in environment variables or .env file")

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)

# --- JWT Token Handling ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a new JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Use the configured expiration time
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, credentials_exception):
    """Verifies a JWT token and returns the username (or raises exception)."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub") # Assuming username is stored in 'sub' claim
        if username is None:
            print("Username missing from token payload.")
            raise credentials_exception
        # Optional: Validate payload against TokenData model if needed
        # token_data = TokenData(username=username)
        return username # Return username directly for simplicity now
    except JWTError as e:
        print(f"JWT Error: {e}")
        raise credentials_exception
    except ValidationError as e: # Handle Pydantic validation error if TokenData is used
         print(f"Token data validation error: {e}")
         raise credentials_exception
    except Exception as e:
         print(f"Unexpected error verifying token: {e}")
         raise credentials_exception

# --- End of auth_utils.py ---
