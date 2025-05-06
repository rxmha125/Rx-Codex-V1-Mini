# main.py (Basic FastAPI Application)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config # Added GPT2Config for safety
import os
import logging
from contextlib import asynccontextmanager

# --- Configuration ---
# Make sure to replace with your actual Hub repo ID if different
HF_REPO_ID = "rxmha125/RxCodexV1-mini" # <<< YOUR HUGGING FACE REPO ID
# Automatically detect CUDA or default to CPU
MODEL_LOAD_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Variables ---
tokenizer = None
model = None
app_config = {"model_repo_id": HF_REPO_ID, "device": MODEL_LOAD_DEVICE} # Store config

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
            else:
                logger.warning("Tokenizer needs a pad token for generation. Consider adding one or ensuring consistency.")
                # Handling missing pad token might be needed depending on model requirements

        logger.info(f"Attempting to load model from {app_config['model_repo_id']} to device {app_config['device']}...")
        model = AutoModelForCausalLM.from_pretrained(app_config['model_repo_id'])
        model.to(app_config['device'])
        model.eval() # Set to evaluation mode
        logger.info(f"Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model or tokenizer on startup: {e}", exc_info=True)
        # In a real app, you might raise an error here to prevent startup
        # For now, model/tokenizer will remain None if loading fails
    yield
    # Cleanup on shutdown
    logger.info("API Shutting down.")
    model = None
    tokenizer = None

app = FastAPI(title="Rx Codex V1-mini API", lifespan=lifespan)

# --- Pydantic Models for Request/Response ---
class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str

# --- API Endpoints ---
@app.get("/")
async def root():
    """ Basic API status endpoint """
    return {"message": f"Rx Codex V1-mini API ({app_config['model_repo_id']}) is running!", "model_status": "Loaded" if model and tokenizer else "Not Loaded"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text_api(request: GenerationRequest):
    """ Generates text based on a given prompt using the loaded Rx Codex model """
    if not tokenizer or not model:
        logger.error("Model or tokenizer not loaded, cannot generate.")
        raise HTTPException(status_code=503, detail="Model is not ready. Please check logs.") # Service Unavailable

    logger.info(f"Received generation request for prompt: '{request.prompt}'")

    try:
        # Determine max length for tokenizer based on model config and requested new tokens
        # Use n_positions from model config if available, default otherwise
        max_length_tokenizer = getattr(model.config, 'n_positions', 512) - request.max_new_tokens
        if max_length_tokenizer <= 0:
            raise HTTPException(status_code=400, detail=f"max_new_tokens ({request.max_new_tokens}) too large for model context window.")

        inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length_tokenizer)
        input_ids = inputs["input_ids"].to(app_config['device'])
        attention_mask = inputs["attention_mask"].to(app_config['device'])

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + request.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Simple greedy search by default
            )

        # Decode only the newly generated tokens (optional, can decode whole sequence)
        # generated_ids = output_sequences[:, input_ids.shape[1]:] # Get only generated token ids
        # generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Decode the whole sequence for simplicity now
        full_generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        logger.info(f"Successfully generated response.")

        # Return the original prompt and the full generated text
        return GenerationResponse(prompt=request.prompt, generated_text=full_generated_text)

    except Exception as e:
        logger.error(f"Error during text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during text generation.")

# --- Uvicorn runner (for convenience if run directly) ---
# Note: Usually you run via 'uvicorn main:app --reload' in terminal
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API via Uvicorn (direct script run)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)