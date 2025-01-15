from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

#load the trained model
model = joblib.load("iris_model.pkl")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisPrediction(BaseModel):
    predicted_class: int
    predicted_class_name: str


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=IrisPrediction)
def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    input_data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )
    # make a prediction
    predicted_class = model.predict(input_data)[0]
    predicted_class_name = load_iris().target_names[predicted_class]

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "predicted_class": predicted_class,
            "predicted_class_name": predicted_class_name,
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_width": petal_width,
            "petal_length": petal_length
        },
    )

