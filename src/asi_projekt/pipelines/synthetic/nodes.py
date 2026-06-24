import logging
import os
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

load_dotenv()

logger = logging.getLogger(__name__)


def _prepare_real_data(real_data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Porządkuje dane rzeczywiste przed użyciem w SDV.

    Normalizuje nazwy kolumn (SQLAlchemy quoted_name -> str), opcjonalnie usuwa
    kolumny wskazane w `drop_columns` (np. identyfikatory i wolny tekst, które
    pogarszają jakość danych syntetycznych) oraz wymusza typy numeryczne tam,
    gdzie to możliwe.

    Args:
        real_data: Surowe dane wczytane z bazy.
        parameters: Sekcja params:synthetic z parameters.yml.

    Returns:
        Oczyszczony DataFrame gotowy dla synthesizera.
    """
    df = real_data.copy()
    df.columns = [str(c) for c in df.columns]

    drop_cols = parameters.get("drop_columns", [])
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        logger.info("SDV: usunięto kolumny przed generowaniem: %s", existing_drop)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    logger.info("SDV: dane wejściowe %d wierszy, %d kolumn", *df.shape)
    return df


def generate_synthetic_data(
    real_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Generuje dane syntetyczne synthesizerem GaussianCopula.

    Args:
        real_data: Dane rzeczywiste wczytane z katalogu (tabela SQLite).
        parameters: Sekcja params:synthetic z parameters.yml.

    Returns:
        Wygenerowany zbiór syntetyczny.
    """
    df = _prepare_real_data(real_data, parameters)

    metadata = Metadata.detect_from_dataframe(df)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)

    synthetic = synthesizer.sample(num_rows=parameters["n_samples"])
    logger.info("SDV: wygenerowano %d rekordów syntetycznych", len(synthetic))
    return synthetic


def evaluate_synthetic_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Dict[str, float]:
    """Ewaluuje jakość danych syntetycznych i loguje wyniki do W&B.

    Liczy dwa raporty SDV:
      - `run_diagnostic` — poprawność struktury (oczekiwane ~1.0),
      - `evaluate_quality` — podobieństwo statystyczne rozkładów (<1.0).

    Oba score'y (przez `.get_score()`) są logowane do Weights & Biases.

    Args:
        real_data: Dane rzeczywiste.
        synthetic_data: Dane wygenerowane przez SDV.
        parameters: Sekcja params:synthetic z parameters.yml.

    Returns:
        Słownik: diagnostic_score, quality_score.
    """
    df = _prepare_real_data(real_data, parameters)

    metadata = Metadata.detect_from_dataframe(df)

    diagnostic = run_diagnostic(df, synthetic_data, metadata)
    quality = evaluate_quality(df, synthetic_data, metadata)

    scores = {
        "diagnostic_score": float(diagnostic.get_score()),  # ~1.0 (struktura)
        "quality_score": float(quality.get_score()),        # <1.0 (podobieństwo)
    }

    logger.info(
        "SDV: diagnostic_score=%.4f | quality_score=%.4f",
        scores["diagnostic_score"], scores["quality_score"],
    )

    try:
        import wandb

        with wandb.init(
            project=os.getenv("WANDB_PROJECT", parameters.get("wandb_project", "asi-projekt")),
            entity=os.getenv("WANDB_ENTITY", parameters.get("wandb_entity")),
            job_type="sdv_evaluation",
            name="sdv-gaussian-copula",
            tags=["sdv", "synthetic", "gaussian-copula"],
            config={"n_samples": len(synthetic_data)},
            reinit=True,
        ):
            wandb.log({
                "sdv/diagnostic_score": scores["diagnostic_score"],
                "sdv/quality_score": scores["quality_score"],
            })
        logger.info("SDV: score'y zalogowane do W&B.")
    except Exception as exc:  # pragma: no cover - logowanie opcjonalne offline
        logger.warning("Nie udało się zalogować do W&B: %s", exc)

    return scores
