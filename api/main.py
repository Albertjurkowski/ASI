import logging
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from .model_loader import load_model
from .schemas import HealthResponse, PredictionResponse, SpaceshipFeatures

logger = logging.getLogger(__name__)

models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ładuje model raz przy starcie; zwalnia zasoby przy zamknięciu."""
    model, model_type = load_model()
    if model is not None:
        models["predictor"] = model
        models["type"] = model_type
    yield
    models.clear()


app = FastAPI(
    title="Spaceship Titanic Prediction API",
    description="REST API do predykcji transportu pasażerów — sprint 5 ASI",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Sprawdza status API i dostępność modelu."""
    return HealthResponse(
        status="ok",
        model_loaded="predictor" in models,
        model_type=models.get("type", "none"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: SpaceshipFeatures) -> PredictionResponse:
    """Zwraca predykcję transportu dla podanych cech pasażera.

    Przyjmuje cechy po preprocessingu (zakodowane numerycznie).
    Zwraca 503 gdy model nie jest załadowany, 500 przy błędzie predykcji.
    """
    if "predictor" not in models:
        raise HTTPException(status_code=503, detail="Model nie jest załadowany. Uruchom najpierw kedro run.")

    try:
        input_df = pd.DataFrame([features.model_dump()])
        model = models["predictor"]
        model_type = models.get("type", "unknown")
        model_name = model_type

        # Try to use predict_proba to get the probability of being transported
        if model_type == "autogluon":
            try:
                model_name = model.get_model_best()
            except Exception:
                pass
            
            try:
                prob = model.predict_proba(input_df)
                if isinstance(prob, pd.DataFrame):
                    # Check columns for class 1 or True
                    if 1 in prob.columns:
                        value = float(prob[1].iloc[0])
                    elif True in prob.columns:
                        value = float(prob[True].iloc[0])
                    else:
                        value = float(prob.iloc[0, 1])
                else:
                    value = float(prob.iloc[0])
            except Exception as exc:
                logger.warning("Błąd AutoGluon predict_proba: %s. Używam predict.", exc)
                prediction = model.predict(input_df)
                value = float(prediction.iloc[0] if hasattr(prediction, "iloc") else prediction[0])
        else:
            if hasattr(model, "predict_proba"):
                try:
                    prob = model.predict_proba(input_df)
                    # For binary classification, class 1 is at index 1
                    value = float(prob[0][1])
                except Exception as exc:
                    logger.warning("Błąd predict_proba: %s. Używam predict.", exc)
                    prediction = model.predict(input_df)
                    value = float(prediction.iloc[0] if hasattr(prediction, "iloc") else prediction[0])
            else:
                prediction = model.predict(input_df)
                value = float(prediction.iloc[0] if hasattr(prediction, "iloc") else prediction[0])

    except Exception as exc:
        logger.error("Błąd predykcji: %s", exc)
        raise HTTPException(status_code=500, detail=f"Błąd predykcji: {exc}")

    return PredictionResponse(prediction=value, model=model_name)
