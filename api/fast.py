
import os
import joblib
from fastapi import fastAPI
from fastapi.middleware.cros import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

path = os.path.dirname(os.path.dirname(__file__))
model = joblib.load(path, "model.jobib")

# Endpoint: Health
@app.get("/")
def index():
    return {"Status": "Up and Running"}

@app.get("/predict-category")
def predict_cat(text: str):

    cat = model.predict([txt])

    return {"category": cat}

@app.get("/predict_probabilities")
def predict_proba(text: str):

    probas = model.predict_proba([txt])
    #change to the equivalent for Ridge model

    return {"probabilities": probas}
