import numpy as np

list_categories = ['appetizers',
                     'artificially sweetened beverages',
                     'biscuits and cakes',
                     'bread',
                     'breakfast cereals',
                     'cereals',
                     'cheese',
                     'chocolate products',
                     'dairy desserts',
                     'dressings and sauces',
                     'dried fruits',
                     'eggs',
                     'fats',
                     'fish and seafood',
                     'fruit juices',
                     'fruit nectars',
                     'fruits',
                     'ice cream',
                     'legumes',
                     'meat',
                     'milk and yogurt',
                     'nuts',
                     'offals',
                     'one dish meals',
                     'pastries',
                     'pizza pies and quiche',
                     'plant based milk substitutes',
                     'potatoes',
                     'processed meat',
                     'salty and fatty products',
                     'sandwiches',
                     'soups',
                     'sweetened beverages',
                     'sweets',
                     'teas and herbal teas and coffees',
                     'unsweetened beverages',
                     'vegetables',
                     'waters and flavored waters']


def get_probabilities_from_ocr(model, text):

    # Load predict probas per category for each OCR
    d = model.decision_function([text])
    probabilities = [np.exp(x) / np.sum(np.exp(d)) for x in d]

    return list(probabilities[0])

def pred_or_hesitate(proba, list_cat):
    decision_threshold = 0.012
    indices_max = np.argsort([-x for x in proba])
    if (proba[indices_max[0]] - proba[indices_max[1]]) > decision_threshold:
        return f"J'ai trouvé ! La catégorie est '{list_cat[indices_max[0]]}'"
    return f"J'hésite entre '{list_cat[indices_max[0]]}' et '{list_cat[indices_max[1]]}'"

