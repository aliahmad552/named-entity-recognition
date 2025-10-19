from fastapi import FastAPI,Request
import os
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model_loader import predict_sentence,model,index2tag,word2index

app = FastAPI(
    title="BiLSTM Named Entity Recognition API",
    description="NER using BiLSTM + Softmax | Developed by Ali Ahmad",
    version="1.0"
)

# ---------- CORS Setup ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify e.g. ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Template Setup ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this is /app folder
PROJECT_DIR = os.path.dirname(BASE_DIR)                # go one level up
TEMPLATES_DIR = os.path.join(PROJECT_DIR, "templates") # points to /named_entity_recognition/templates
INDEX_FILE = os.path.join(TEMPLATES_DIR, "index.html")

class InputText(BaseModel):
    text: str
    debug: bool = False   # optional flag to enable console debug

@app.get("/")
async def serve_ui():
    """Serve static index.html without Jinja."""
    return FileResponse(INDEX_FILE)

@app.post("/predict/")
def predict_entities(input_data: InputText):
    text = input_data.text
    entities = predict_sentence(text, model, word2index, index2tag)
    return {"text": text, "entities": entities}
