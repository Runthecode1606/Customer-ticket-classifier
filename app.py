import time
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Customer Support Ticket Classifier")

# Load model and tokenizer from your local folder
MODEL_PATH = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

class TicketRequest(BaseModel):
    text: str

class TicketResponse(BaseModel):
    category: str
    confidence: float
    inference_latency_ms: float

@app.post("/predict", response_model=TicketResponse)
def predict_ticket(request: TicketRequest):
    start_time = time.time()
    
    # Tokenize input text
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, class_idx = torch.max(probs, dim=-1)
        
    latency = (time.time() - start_time) * 1000
    
    return TicketResponse(
        category=model.config.id2label[class_idx.item()],
        confidence=float(confidence.item()),
        inference_latency_ms=latency
    )