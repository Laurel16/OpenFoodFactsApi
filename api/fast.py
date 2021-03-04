
import os
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

#@app.get("/predict_probabilities")
#def predict_proba(text: str):

    #probas = model.predict_proba([text])
    #change to the equivalent for Ridge model

    #return {"probabilities": probas}
