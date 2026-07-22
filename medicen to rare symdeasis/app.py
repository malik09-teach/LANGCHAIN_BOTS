from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import markdown
import os

# Import our backend logic
from models import PatientState, Demographics
from orchestrator import handle_turn

app = FastAPI(title="Symptom-to-Rare-Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory to simulate sessions (for prototype only)
SESSIONS = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if req.session_id not in SESSIONS:
        SESSIONS[req.session_id] = {
            "patient_state": PatientState(
                patient_id=str(uuid.uuid4()),
                demographics=Demographics()
            ),
            "hypotheses": [],
            "turn": 1
        }
        
    session = SESSIONS[req.session_id]
    
    # Run the orchestrator
    response_text = handle_turn(
        user_text=req.message,
        patient_state=session["patient_state"],
        hypotheses=session["hypotheses"],
        turn=session["turn"]
    )
    
    session["turn"] += 1
    
    # Convert markdown response to HTML for easier rendering in UI
    html_response = markdown.markdown(response_text)
    
    return JSONResponse({
        "response": html_response,
        "raw_response": response_text
    })

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
