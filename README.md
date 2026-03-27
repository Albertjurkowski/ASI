# ASI Projekt — Spaceship Titanic

Projekt klasyfikacji binarnej na danych z konkursu Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic). Celem jest przewidzenie czy pasażer został przetransportowany do innego wymiaru (`Transported`: True/False).

Pipeline zbudowany w Kedro 0.19.9.

## Dane

Dane pobrane z Kaggle, załadowane do bazy SQLite (`data/01_raw/dataset.db`). Tabela: `spaceship_titanic`, ~8700 wierszy, 14 kolumn.

## Model

Baseline: `RandomForestClassifier` (100 drzew, random_state=42).

Wyniki na zbiorze walidacyjnym:
- Accuracy: 0.7807
- F1 Score: 0.7776

## Struktura projektu

```
asi_projekt/
├── conf/
│   ├── base/
│   │   ├── catalog.yml        # definicje źródeł danych
│   │   └── parameters.yml     # parametry pipeline'u
│   └── local/
│       └── credentials.yml    # dane do bazy (gitignored)
├── data/
│   ├── 01_raw/                # surowe dane i baza SQLite
│   ├── 06_models/             # wytrenowany model
│   └── 08_reporting/          # metryki JSON
├── notebooks/
│   └── 01_eda.ipynb           # EDA i baseline z fazy 1
└── src/
    └── asi_projekt/
        └── pipelines/
            └── data_processing/
                ├── nodes.py       # funkcje przetwarzania
                └── pipeline.py    # definicja pipeline'u
```

## Uruchomienie

```bash
# aktywuj środowisko
.venv\Scripts\activate

# uruchom pipeline
kedro run

# sprawdź metryki
type data\08_reporting\metrics.json
```

## Pipeline

4 node'y wykonywane w kolejności:

1. `preprocess_node` — usuwa zbędne kolumny, koduje kategoryczne, uzupełnia braki
2. `split_data_node` — podział 70/15/15 (train/val/test)
3. `train_model_node` — trening RandomForest
4. `evaluate_model_node` — obliczenie metryk na zbiorze walidacyjnym