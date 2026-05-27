import logging
import os
from typing import Any, Dict, Tuple

import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

load_dotenv()

logger = logging.getLogger(__name__)


def preprocess(
        data: pd.DataFrame,
        parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Przetwarza surowe dane wejściowe.

    Usuwa niepotrzebne kolumny, koduje zmienne kategoryczne
    i uzupełnia brakujące wartości medianą.

    Args:
        data: Surowe dane z bazy.
        parameters: Parametry z pliku parameters.yml.

    Returns:
        Oczyszczony DataFrame gotowy do dalszego przetwarzania.
    """
    target = parameters["target_column"]
    drop_cols = parameters.get("drop_columns", ["PassengerId", "Name", "Cabin"])

    df = data.copy()

    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop)
    logger.info("Usunięto kolumny: %s", existing_drop)

    df[target] = df[target].map({True: 1, False: 0, "True": 1, "False": 0})

    cat_cols = [
        c for c in df.select_dtypes(include=["object", "bool"]).columns
        if c != target
    ]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    logger.info("Zakodowano kolumny kategoryczne: %s", cat_cols)

    n_missing_before = df.isnull().sum().sum()
    df = df.fillna(df.median(numeric_only=True))
    logger.info("Uzupełniono braków: %d wartości medianą", n_missing_before)

    logger.info("Po preprocessingu: %d wierszy, %d kolumn", *df.shape)
    return df


def split_data(
        data: pd.DataFrame,
        parameters: Dict[str, Any],
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    """Dzieli dane na zbiory treningowy, walidacyjny i testowy.

    Podział odbywa się w proporcji 70/15/15 zgodnie z parametrami.

    Args:
        data: Przetworzony DataFrame z kolumną docelową.
        parameters: Parametry zawierające target_column oraz ustawienia podziału.

    Returns:
        Krotka zawierająca X_train, X_val, X_test, y_train, y_val, y_test.
    """
    target = parameters["target_column"]
    split_params = parameters["split"]

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=split_params["val_ratio"],
        random_state=split_params["random_state"],
    )

    logger.info(
        "Podział: Train=%d | Val=%d | Test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        parameters: Dict[str, Any],
) -> RandomForestClassifier:
    """Trenuje model klasyfikacji Random Forest.

    Parametry modelu pobierane są z pliku parameters.yml,
    dzięki czemu nie ma hardcodowanych wartości w kodzie.

    Args:
        X_train: Cechy zbioru treningowego.
        y_train: Etykiety zbioru treningowego.
        parameters: Parametry zawierające konfigurację modelu.

    Returns:
        Wytrenowany model RandomForestClassifier.
    """
    model_params = parameters["model"]
    clf = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params.get("max_depth"),
        random_state=model_params["random_state"],
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    logger.info(
        "Model wytrenowany: %d drzew, max_depth=%s, random_state=%d",
        model_params["n_estimators"],
        model_params.get("max_depth"),
        model_params["random_state"],
    )
    return clf


def evaluate_and_log(
        model: RandomForestClassifier,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        parameters: Dict[str, Any],
) -> Dict[str, float]:
    """Ewaluuje model na zbiorze walidacyjnym i loguje wyniki do W&B.

    Loguje do W&B: konfigurację eksperymentu (config), metryki
    klasyfikacji binarnej, wykres ważności cech, macierz pomyłek
    oraz artefakt z wytrenowanym modelem.

    Args:
        model: Wytrenowany klasyfikator scikit-learn.
        X_val: Cechy zbioru walidacyjnego.
        y_val: Etykiety zbioru walidacyjnego.
        parameters: Cały słownik parameters.yml — logowany jako config.

    Returns:
        Słownik z metrykami ewaluacji (accuracy, f1, precision, recall).
    """
    model_params = parameters["model"]
    split_params = parameters["split"]

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "asi-projekt"),
        entity=os.getenv("WANDB_ENTITY"),  # None => personal workspace
        name=f"rf-n{model_params['n_estimators']}-d{model_params.get('max_depth')}",
        config={
            "model_type": "RandomForestClassifier",
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params.get("max_depth"),
            "random_state": model_params["random_state"],
            "test_size": split_params["test_size"],
            "val_ratio": split_params["val_ratio"],
            "target_column": parameters["target_column"],
        },
        tags=["baseline", "sklearn", "classification"],
        reinit=True,
    )

    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "f1_score": round(float(f1_score(y_val, y_pred)), 4),
        "precision": round(float(precision_score(y_val, y_pred)), 4),
        "recall": round(float(recall_score(y_val, y_pred)), 4),
    }

    wandb.log(metrics)

    if hasattr(model, "feature_importances_"):
        try:
            wandb.sklearn.plot_feature_importances(
                model, feature_names=list(X_val.columns)
            )
        except Exception as exc:  # pragma: no cover - opcjonalny wykres
            logger.warning("Nie udało się zalogować feature importances: %s", exc)

    try:
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_val.tolist(),
                preds=y_pred.tolist(),
                class_names=["Not Transported", "Transported"],
            )
        })
    except Exception as exc:  # pragma: no cover - opcjonalny wykres
        logger.warning("Nie udało się zalogować confusion matrix: %s", exc)

    model_path = "data/06_models/baseline_model.pkl"
    if os.path.exists(model_path):
        artifact = wandb.Artifact(
            name="baseline-model",
            type="model",
            description=(
                f"RandomForestClassifier n={model_params['n_estimators']}, "
                f"max_depth={model_params.get('max_depth')}"
            ),
        )
        artifact.add_file(model_path)
        run.log_artifact(artifact)
    else:
        logger.warning("Plik modelu %s nie istnieje - pomijam artefakt", model_path)

    wandb.finish()

    logger.info(
        "W&B: run zalogowany. Accuracy=%.4f | F1=%.4f | Precision=%.4f | Recall=%.4f",
        metrics["accuracy"],
        metrics["f1_score"],
        metrics["precision"],
        metrics["recall"],
    )
    return metrics
