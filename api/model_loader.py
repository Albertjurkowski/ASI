import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
AUTOGLUON_PATH = BASE_DIR / "data" / "06_models" / "autogluon"
BASELINE_PATH = BASE_DIR / "data" / "06_models" / "baseline_model.pkl"


def load_model() -> tuple[Any, str]:
    """Wczytuje najlepszy dostępny model (AutoGluon → baseline RF).

    Returns:
        Krotka (model, typ_modelu). Jeśli żaden model nie jest dostępny,
        zwraca (None, "none").
    """
    if AUTOGLUON_PATH.exists():
        try:
            from autogluon.tabular import TabularPredictor
            predictor = TabularPredictor.load(str(AUTOGLUON_PATH))
            logger.info("Załadowano AutoGluon z %s", AUTOGLUON_PATH)
            return predictor, "autogluon"
        except Exception as exc:
            logger.warning("Nie udało się załadować AutoGluon: %s", exc)

    if BASELINE_PATH.exists():
        with open(BASELINE_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("Załadowano baseline RF z %s", BASELINE_PATH)
        return model, "baseline"

    logger.error("Brak dostępnych modeli w %s ani %s", AUTOGLUON_PATH, BASELINE_PATH)
    return None, "none"
