# Rx Codex V1-mini API

This repository contains the FastAPI application to serve the Rx Codex V1-mini language model (~45M parameters, trained from scratch for ~173k steps).

## Model

The model served by this API can be found on Hugging Face Hub:
[rxmha125/RxCodexV1-mini](https://huggingface.co/rxmha125/RxCodexV1-mini)

**Note:** This model is currently experimental and has significant limitations due to its size and training process. Expect basic pattern matching, repetition, and potential incoherence.

## Setup (Local)

1.  Clone this repository:
    ```bash
    git clone <your-repo-url>
    cd rx_codex_api
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # Linux/macOS: source venv/bin/activate
    ```
3.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have a compatible PyTorch version installed if using GPU)*

## Running Locally

Run the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. The model and tokenizer will be downloaded from Hugging Face Hub on the first startup.

## API Endpoint

### Generate Text

* **URL:** `/generate`
* **Method:** `POST`
* **Request Body (JSON):**
    ```json
    {
      "prompt": "Your input text here",
      "max_new_tokens": 50
    }
    ```
* **Response Body (JSON):**
    ```json
    {
      "prompt": "Your input text here",
      "generated_text": "Model's generated output..."
    }
    ```
* **Example using `curl`:**
    ```bash
    curl -X 'POST' \
      'http://localhost:8000/generate' \
      -H 'Content-Type: application/json' \
      -d '{
        "prompt": "Hello there!",
        "max_new_tokens": 30
      }'
    ```

## Deployment

*(Add details here later about Vercel deployment)*