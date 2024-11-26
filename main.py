from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from preprocessor import SuperPreprocessor
import numpy as np

with open("data/trained_pipeline.pkl", "rb") as f:
    trained_pipeline = pickle.load(f)

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def get_one_x(item: Item):
    df = pd.DataFrame([item.model_dump()])
    X = df.drop('selling_price', axis=1)

    return X

def get_many_x(items: List[Item]):
    df = pd.DataFrame([item.model_dump() for item in items])
    X = df.drop('selling_price', axis=1)

    return X

@app.get("/")
def predict_item(test_key: str) -> str:
    return f"Hi there! It's a test endpoint. You passes a key: {test_key}"

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    X = get_one_x(item)
    predictions = trained_pipeline.predict(X)

    return predictions


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    X = get_many_x(items)
    predictions = trained_pipeline.predict(X)
    
    return predictions