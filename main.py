# main.py (Corrected for Render PORT and previous fixes)

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
import os
import logging
import time
import re # Ensure re is imported
from contextlib import asynccontextmanager
from typing import Optional, List # Ensure Optional/List are imported
from datetime import timedelta

# --- Import our API modules ---
try:
    from api import db, models, auth_utils
except ImportError:
    print("Error importing local api modules. Ensure 'api' folder exists and contains db.py, models.py, auth_utils.py.")
    raise

# --- Configuration ---
HF_REPO_ID = "rxmha125/RxCodexV1-mini" # <<< YOUR HUGGING FACE REPO ID
MODEL_LOAD_DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Detect device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Variables & Lifespan (Model Loading) ---
tokenizer = None
model = None
app_config = {"model_repo_id": HF_REPO_ID, "device": MODEL_LOAD_DEVICE}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, app_config
    logger.info(f"API Startup: Loading resources...")
    logger.info(f"Attempting to load tokenizer from {app_config['model_repo_id']}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(app_config['model_repo_id'])
        if tokenizer.pad_token is None and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded.")
    except Exception as e: logger.error(f"FATAL: Tokenizer loading failed: {e}", exc_info=True)

    logger.info(f"Attempting to load model from {app_config['model_repo_id']} to device {app_config['device']}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(app_config['model_repo_id'])
        model.to(app_config['device'])
        model.eval() # Set to evaluation mode
        logger.info(f"Model loaded successfully.")
        app_config['max_seq_len'] = getattr(model.config, 'n_positions', 256)
    except Exception as e: logger.error(f"FATAL: Model loading failed: {e}", exc_info=True)

    if model and tokenizer: logger.info("Model and tokenizer loaded successfully.")
    else: logger.warning("Model or tokenizer failed to load on startup.")
    yield
    logger.info("API Shutting down.")
    model = None; tokenizer = None

app = FastAPI(title="Rx Codex V1-mini API", lifespan=lifespan)

# --- Pydantic Models ---
class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    execution_time_sec: Optional[float] = None

# --- Authentication Dependencies ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Authentication Endpoints ---
@app.post("/users/register", response_model=models.UserPublic, status_code=status.HTTP_201_CREATED)
async def register_user(user: models.UserCreate):
    logger.info(f"Attempting registration for username: {user.username}")
    user_collection = db.get_user_collection()
    if user_collection is None: # Correct check
         logger.error("Registration failed: Database collection not available.")
         raise HTTPException(status_code=503, detail="Database service not available")

    existing_user = db.get_user(user.username) # Uses corrected check internally
    if existing_user:
        logger.warning(f"Registration failed: Username '{user.username}' already exists.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")

    hashed_password = auth_utils.get_password_hash(user.password)
    user_in_db = models.UserInDB(username=user.username, email=user.email, hashed_password=hashed_password)
    try:
         new_user_result = user_collection.insert_one(user_in_db.model_dump())
         logger.info(f"User '{user.username}' registered successfully with ID: {new_user_result.inserted_id}")
         return models.UserPublic(username=user_in_db.username, email=user_in_db.email)
    except Exception as e:
         logger.error(f"Database error during registration for {user.username}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Could not register user due to server error")


@app.post("/token", response_model=models.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    logger.info(f"Login attempt for username: {form_data.username}")
    user = db.get_user(form_data.username)
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Login failed for username: {form_data.username} - Incorrect credentials.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_utils.create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    logger.info(f"Login successful for username: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": f"Rx Codex V1-mini API ({app_config.get('model_repo_id', 'N/A')}) is running!", "model_status": "Loaded" if model and tokenizer else "Not Loaded"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text_api(request: GenerationRequest): # Will protect later
    global tokenizer, model
    if not tokenizer or not model: raise HTTPException(status_code=503, detail="Model is not ready.")
    logger.info(f"Received generation request for prompt: '{request.prompt}'")
    start_time = time.monotonic()
    try:
        max_len_model = app_config.get('max_seq_len', 256)
        max_length_tokenizer = max_len_model - request.max_new_tokens
        if max_length_tokenizer <= 0: raise HTTPException(status_code=400, detail="max_new_tokens too large.")
        inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length_tokenizer)
        input_ids = inputs["input_ids"].to(app_config['device'])
        attention_mask = inputs["attention_mask"].to(app_config['device'])
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=input_ids.shape[1] + request.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        full_generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        # --- Basic cleanup of response ---
        assistantContent = full_generated_text
        if assistantContent.lower().startswith(request.prompt.lower()):
             assistantContent = assistantContent[len(request.prompt):].strip()
             # Use correct Python regex substitution
             assistantContent = re.sub(r"^\s*[:\-.,\s]\s*", "", assistantContent)
        assistantContent = assistantContent.strip() or "Model returned an empty response."
        # --- End cleanup ---
    except Exception as e:
        logger.error(f"Error during text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during text generation.")
    finally:
        end_time = time.monotonic()
        execution_time = end_time - start_time
        logger.info(f"Generation complete in {execution_time:.4f} seconds.")
    return GenerationResponse(prompt=request.prompt, generated_text=assistantContent, execution_time_sec=execution_time)

# --- Uvicorn runner ---
if __name__ == "__main__":
    import uvicorn
    # Need to import Optional here if not done globally
    from typing import Optional # Ensure Optional is imported if used here
    logger.info("Starting API via Uvicorn (direct script run)...")
    # Use PORT environment variable if set (Render provides this), otherwise default to 8000
    port_to_use = int(os.getenv("PORT", 8000))
    logger.info(f"Uvicorn will attempt to run on host 0.0.0.0:{port_to_use}")
    # This block is mainly for local execution using `python main.py`
    # Render uses the "Start Command" set in its dashboard.
    # Use reload=True only for local dev, not for Render's start command.
    uvicorn.run("main:app", host="0.0.0.0", port=port_to_use, reload=True)