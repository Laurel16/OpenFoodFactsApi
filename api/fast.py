
import os
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.helpers import pred_or_hesitate, get_probabilities_from_ocr, list_categories

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#path = os.path.dirname(os.path.dirname(__file__))

model = joblib.load("model.joblib")

# Endpoint: Health
@app.get("/")
def index():
    return {"Status": "Up and Running"}



@app.get("/predict-category")
async def predict_cat(text: str):

    cat = model.predict([text])

    return {"category": cat.item(0)}

@app.get("/predict_probabilities")
async def predict_proba(text: str):

    probas = get_probabilities_from_ocr(model, text)

    result = pred_or_hesitate(probas, list_categories)
    #change to the equivalent for Ridge model

    return {"result": result}


