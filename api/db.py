# api/db.py (Corrected Truthiness Check)

from pymongo import MongoClient, errors
from pymongo.collection import Collection
import os
from dotenv import load_dotenv
from typing import Optional

# Import the Pydantic model for type hinting
try:
    from .models import UserInDB
except ImportError:
    from models import UserInDB # Fallback

# Load environment variables from .env file
load_dotenv()

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")
DB_NAME = os.getenv("DB_NAME", "RxCodexDB")
USER_COLLECTION = "users"

if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL not found in .env file.")

# --- Initialize Database Client ---
client: Optional[MongoClient] = None
try:
    client = MongoClient(DATABASE_URL, serverSelectionTimeoutMS=5000)
    client.admin.command('ping') # Verify connection
    print("MongoDB connection successful.")
except errors.ConnectionFailure as e:
    print(f"MongoDB connection failed: Could not connect to server: {e}")
    client = None
except Exception as e:
    print(f"An unexpected error occurred during MongoDB connection: {e}")
    client = None

# --- Database Access Functions ---

def get_db():
    """Returns the database instance if client is connected."""
    if client:
        try:
            # Optional: Re-ping to ensure connection is alive before returning
            # client.admin.command('ping')
            return client[DB_NAME]
        except errors.ConnectionFailure:
            print("Database connection lost. Returning None.")
            return None
        except Exception as e:
            print(f"Error accessing database: {e}")
            return None
    else:
        # print("Database client not initialized.") # Can be noisy
        return None

def get_user_collection() -> Optional[Collection]:
    """Returns the user collection instance."""
    db = get_db()
    # --- *** CORRECTED HERE *** ---
    if db is not None:
    # --- ********************** ---
        try:
            # Optional: Ensure index exists on username for faster lookups and uniqueness
            # try:
            #     db[USER_COLLECTION].create_index("username", unique=True)
            # except errors.OperationFailure: # Handle potential errors if index already exists etc.
            #     pass
            return db[USER_COLLECTION]
        except Exception as e:
             print(f"Error accessing user collection: {e}")
             return None
    else:
        return None

def get_user(username: str) -> Optional[UserInDB]:
    """Fetches a user from the database by username."""
    user_collection = get_user_collection()
    # --- *** CORRECTED HERE *** ---
    if user_collection is not None:
    # --- ********************** ---
        try:
            user_data = user_collection.find_one({"username": username})
            if user_data:
                # Pydantic model initialization ignores extra fields like _id by default
                return UserInDB(**user_data)
            else:
                return None # User not found
        except Exception as e:
            print(f"Error fetching user '{username}': {e}")
            return None
    else:
        return None # Collection not available

