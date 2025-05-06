# main.py (Updated with Execution Time)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
import os
import logging
import time # Import the time module
from contextlib import asynccontextmanager
from typing import Optional # Import Optional for the new field

# --- Configuration ---
HF_REPO_ID = "rxmha125/RxCodexV1-mini" # <<< YOUR HUGGING FACE REPO ID
MODEL_LOAD_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Variables ---
tokenizer = None
model = None
app_config = {"model_repo_id": HF_REPO_ID, "device": MODEL_LOAD_DEVICE}

# --- API Lifespan (Model Loading on Startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    logger.info(f"API Startup: Loading resources...")
    logger.info(f"Attempting to load tokenizer from {app_config['model_repo_id']}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(app_config['model_repo_id'])
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Tokenizer pad_token set to eos_token.")
            else: logger.warning("Tokenizer needs pad token.")

        logger.info(f"Attempting to load model from {app_config['model_repo_id']} to device {app_config['device']}...")
        model = AutoModelForCausalLM.from_pretrained(app_config['model_repo_id'])
        model.to(app_config['device'])
        model.eval() # Set to evaluation mode
        logger.info(f"Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model or tokenizer on startup: {e}", exc_info=True)
    yield
    logger.info("API Shutting down.")
    model = None
    tokenizer = None

app = FastAPI(title="Rx Codex V1-mini API", lifespan=lifespan)

# --- Pydantic Models for Request/Response ---
class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

# --- *** UPDATED Response Model *** ---
class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    execution_time_sec: Optional[float] = None # Added optional field for time
# --- **************************** ---

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": f"Rx Codex V1-mini API ({app_config['model_repo_id']}) is running!", "model_status": "Loaded" if model and tokenizer else "Not Loaded"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text_api(request: GenerationRequest):
    global tokenizer, model

    if not tokenizer or not model:
        logger.error("Model or tokenizer not loaded, cannot generate.")
        raise HTTPException(status_code=503, detail="Model is not ready. Please check logs.")

    logger.info(f"Received generation request for prompt: '{request.prompt}'")

    # --- Start Timer ---
    start_time = time.monotonic()

    try:
        # Determine max length for tokenizer
        max_len_model = getattr(model.config, 'n_positions', 256)
        max_length_tokenizer = max_len_model - request.max_new_tokens
        if max_length_tokenizer <= 0:
            raise HTTPException(status_code=400, detail=f"max_new_tokens ({request.max_new_tokens}) too large.")

        # Prepare inputs
        inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length_tokenizer)
        input_ids = inputs["input_ids"].to(app_config['device'])
        attention_mask = inputs["attention_mask"].to(app_config['device'])

        # Generate text
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + request.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        full_generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        # --- End Timer & Calculate Duration ---
        end_time = time.monotonic()
        execution_time = end_time - start_time
        logger.info(f"Generation complete in {execution_time:.4f} seconds.")

        # Return response including execution time
        return GenerationResponse(
            prompt=request.prompt,
            generated_text=full_generated_text,
            execution_time_sec=execution_time # Include the calculated time
        )

    except Exception as e:
        logger.error(f"Error during text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during text generation.")

# --- Uvicorn runner (for local testing if run directly) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API via Uvicorn (direct script run)...")
    # Note: Render uses its own start command, this block isn't used by Render deployment
    uvicorn.run(app, host="0.0.0.0", port=8000)