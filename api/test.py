
from api.helpers import pred_or_hesitate, get_probabilities_from_ocr, list_categories
import joblib

model=joblib.load("model.joblib")
probas = get_probabilities_from_ocr(model, "farine")
print(pred_or_hesitate(probas, list_categories))
