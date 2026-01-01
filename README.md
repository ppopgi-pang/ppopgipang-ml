# FastAPI Project

This is a basic FastAPI project environment.

## Setup

1.  **Create and activate virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Run

To run the server with hot reload:
```bash
uvicorn main:app --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
Documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
