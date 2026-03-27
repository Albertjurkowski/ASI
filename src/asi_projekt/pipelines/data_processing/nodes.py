import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
        random_state=model_params["random_state"],
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    logger.info(
        "Model wytrenowany: %d drzew, random_state=%d",
        model_params["n_estimators"],
        model_params["random_state"],
    )
    return clf


def evaluate_model(
        model: RandomForestClassifier,
        X_val: pd.DataFrame,
        y_val: pd.Series,
) -> Dict[str, float]:
    """Ewaluuje wytrenowany model na zbiorze walidacyjnym.

    Oblicza podstawowe metryki klasyfikacji binarnej.

    Args:
        model: Wytrenowany klasyfikator.
        X_val: Cechy zbioru walidacyjnego.
        y_val: Etykiety zbioru walidacyjnego.

    Returns:
        Słownik z metrykami accuracy i f1_score.
    """
    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "f1_score": round(float(f1_score(y_val, y_pred)), 4),
    }

    logger.info(
        "Metryki (val): Accuracy=%.4f | F1=%.4f",
        metrics["accuracy"],
        metrics["f1_score"],
    )
    return metrics