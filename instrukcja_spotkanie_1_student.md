# ASI — Spotkanie 1: Wprowadzenie do MLOps

## Informacje organizacyjne

**Przedmiot:** Architektury rozwiązań i metodyki wdrożeń SI (MLOps)
**Forma:** Laboratorium, 8 spotkań × 1,5h (co 2 tygodnie)
**Ocena:** Na podstawie projektu zespołowego (2–3 osoby) realizowanego sprint po sprincie

---

## Zasady kursu

### Forma pracy

Przez cały semestr pracujecie w zespołach nad **jednym projektem ML end-to-end**. Każde spotkanie to review poprzedniego sprintu + wprowadzenie do kolejnego. Między spotkaniami pracujecie samodzielnie.

### Co oceniam

- **Działający system** (40%) — czy pipeline, API i dashboard działają na żywo
- **Jakość kodu** (20%) — type hints, docstringi, konfiguracja, brak danych uwierzytelniających w kodzie
- **Eksperymenty ML** (20%) — śledzenie eksperymentów, porównanie modeli
- **Dokumentacja** (20%) — README, diagram architektury, instrukcja uruchomienia

### Wymagania jakościowe (od sprintu 2)

- Type hints w funkcjach
- Docstringi (opis co funkcja robi, parametry, zwracana wartość)
- Parametry w plikach YAML — **nigdy** hardcoded w kodzie
- Dane uwierzytelniające (klucze API, hasła) **nigdy** w repozytorium — zawsze `.env` lub `conf/local/`

---

## Architektura docelowa

Na koniec semestru Wasz projekt będzie wyglądał tak:

```txt
[Źródło danych (SQLite)]
        │
        ▼
[Kedro Pipeline]  ←  parametry z YAML
   ├── ładowanie danych
   ├── preprocessing
   ├── trening modelu (scikit-learn + AutoGluon)
   ├── ewaluacja
   └── generowanie danych syntetycznych (SDV)
        │
        ├──→ [Weights & Biases]  (śledzenie eksperymentów)
        │
        ▼
[FastAPI — REST API]  ←  model załadowany z artefaktu
   ├── POST /predict
   ├── GET /health
   └── GET /model-info
        │
        ▼
[Streamlit — Dashboard]
   ├── Formularz predykcji
   ├── Podgląd danych
   └── Dane syntetyczne
```

### Stos technologiczny

| Warstwa                 | Narzędzie          | Rola                                  |
|-------------------------|--------------------|---------------------------------------|
| Dane                    | SQLite             | Baza danych                           |
| Dane syntetyczne        | SDV                | Generowanie danych syntetycznych      |
| Pipeline                | Kedro              | Orkiestracja, struktura projektu      |
| Modelowanie             | AutoGluon          | AutoML                                |
| Śledzenie eksperymentów | Weights & Biases   | Metryki, porównanie modeli, artefakty |
| API                     | FastAPI + Pydantic | Serwowanie modelu, walidacja danych   |
| Dashboard               | Streamlit          | Interfejs użytkownika                 |
| Środowisko              | Conda              | Izolacja zależności                   |

---

## Harmonogram sprintów

| Sprint | Temat                                          | Deadline    |
|--------|------------------------------------------------|-------------|
| 1      | Notebook + SQLite + EDA + baseline model       | Spotkanie 2 |
| 2      | Modularyzacja kodu + Kedro pipeline            | Spotkanie 3 |
| 3      | Weights & Biases (śledzenie eksperymentów)     | Spotkanie 4 |
| 4      | AutoGluon (AutoML)                             | Spotkanie 5 |
| 5      | FastAPI + Pydantic (REST API)                  | Spotkanie 6 |
| 6      | Streamlit + SDV (dashboard + dane syntetyczne) | Spotkanie 7 |
| 7      | Integracja + dokumentacja                      | Spotkanie 8 |
| 8      | Prezentacje końcowe                            | —           |

---

## Setup środowiska

### 1. Zainstaluj Minicondę (jeśli nie masz)

**macOS (Apple Silicon):**

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

**macOS (Intel):**

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

**Windows:**
Pobierz instalator z [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

**Linux:**

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Po instalacji zrestartuj terminal i sprawdź:

```bash
conda --version
```

### 2. Stwórz środowisko

```bash
conda create -n asi python=3.10 -y
conda activate asi
```

### 3. Zainstaluj zależności

```bash
pip install scikit-learn pandas matplotlib seaborn
pip install jupyter notebook
pip install python-dotenv
```

> Pozostałe narzędzia (Kedro, W&B, AutoGluon, FastAPI, Streamlit, SDV) zainstalujemy w kolejnych sprintach, kiedy będą potrzebne.

### 4. Zainicjalizuj repozytorium Git

```bash
mkdir asi-projekt
cd asi-projekt
git init
```

### 5. Stwórz `.gitignore`

```txt
# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# Środowisko
.env
*.egg-info/

# Dane i modele
data/
*.db
*.pkl
AutogluonModels/

# W&B
wandb/

# IDE
.vscode/
.idea/

# macOS
.DS_Store
```

### 6. Skonfiguruj zarządzanie danymi uwierzytelniającymi

Stwórz plik `.env.example` (ten commitujemy — to szablon):

```txt
WANDB_API_KEY=your_key_here
WANDB_PROJECT=asi-projekt
DATABASE_PATH=data/01_raw/dataset.db
```

Stwórz plik `.env` (kopia `.env.example` z prawdziwymi wartościami — **ten NIE commitujemy**, jest w `.gitignore`):

```bash
cp .env.example .env
```

> **WAŻNE:** Nigdy nie commitujcie kluczy API, haseł ani tokenów do repozytorium Git. Nawet jeśli potem je usuniecie — zostają w historii. Używajcie `.env` + `.gitignore`.

---

## SPRINT 1: Notebook + SQLite + EDA + baseline model

### Cel

Wybrać problem ML, załadować dane do SQLite, przeprowadzić eksplorację danych (EDA), wytrenować baseline model.

### Wybór problemu ML i zbioru danych

Każdy zespół wybiera **inny** problem i zbiór danych. Wymagania:

- Dane tabelaryczne (nie obrazy, nie tekst)
- Minimum 1000 obserwacji
- Problem klasyfikacji lub regresji
- Dane publicznie dostępne

**Propozycje** (możecie wybrać własny):

# Propozycje zbiorów danych do wykorzystania w projekcie

| Zbiór                             | Typ              | Źródło     | Główny plik / CSV                                                                                                         |
|-----------------------------------|------------------|------------|---------------------------------------------------------------------------------------------------------------------------|
| Spaceship Titanic                 | Klasyfikacja     | Kaggle     | `train.csv`, `test.csv`: https://www.kaggle.com/competitions/spaceship-titanic                                            |
| Used Car Prices                   | Regresja         | Kaggle     | `cars_data.csv`: https://www.kaggle.com/datasets/mohammedadham45/cars-data                                                |
| Heart Failure Prediction          | Klasyfikacja     | UCI        | `heart_failure_clinical_records_dataset.csv`: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records      |
| Student Performance               | Regresja/klasyf. | UCI        | `student-mat.csv`, `student-por.csv`: https://archive.ics.uci.edu/dataset/320/student+performance                         |
| Airline Passenger Satisfaction    | Klasyfikacja     | Kaggle     | `train.csv`, `test.csv`: https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction                    |
| House Prices (Ames Iowa)          | Regresja         | Kaggle     | `train.csv`, `test.csv`: https://www.kaggle.com/c/house-prices-advanced-regression-techniques                             |
| Employee Attrition (IBM HR)       | Klasyfikacja     | Kaggle     | `WA_Fn-UseC_-HR-Employee-Attrition.csv`: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset |
| Energy Efficiency                 | Regresja         | UCI        | `ENB2012_data.xlsx` / wersje CSV: https://archive.ics.uci.edu/ml/datasets/energy+efficiency                               |
| Online Shoppers Intention         | Klasyfikacja     | UCI        | `online_shoppers_intention.csv`: https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset     |
| Bike Sharing Demand               | Regresja         | Kaggle     | `train.csv`, `test.csv`: https://www.kaggle.com/c/bike-sharing-demand                                                     |
| Predykcja cukrzycy (Pima Indians) | Klasyfikacja     | UCI/Kaggle | UCI: `diabetes`; Kaggle: `diabetes.csv`: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database             |
| Zdatność wody (Water Potability)  | Klasyfikacja     | Kaggle     | `water_potability.csv`: https://www.kaggle.com/datasets/adityakadiwal/water-potability                                    |
| Klasyfikacja grzybów (Mushroom)   | Klasyfikacja     | UCI/Kaggle | `mushrooms.csv`: https://www.kaggle.com/datasets/uciml/mushroom-classification                                            |
| Bank churn (Customer Churn)       | Klasyfikacja     | Kaggle     | `Churn_Modelling.csv`: https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers                       |


### Deliverables

1. **Repozytorium Git** z poprawnym `.gitignore`
2. **Baza SQLite** z załadowanymi danymi w `data/01_raw/`
3. **Notebook** `notebooks/01_eda.ipynb` zawierający:
   - Ładowanie danych z SQLite
   - Statystyki opisowe (`df.describe()`)
   - Minimum 5 wizualizacji (rozkłady cech, korelacje, zależności)
   - Identyfikacja braków danych i duplikatów
   - Podział na zbiory: train (70%) / val (15%) / test (15%)
   - Baseline model (RandomForest lub LogisticRegression)
   - Metryki ewaluacji na zbiorze walidacyjnym
4. **Plik `requirements.txt`** lub `environment.yml`
5. **Plik `.env.example`** (szablon bez wartości) + `.env` w `.gitignore`

---

### Kryteria oceny sprintu 1

| Kryterium                                                       | Wymagane |
|-----------------------------------------------------------------|----------|
| Dane w SQLite (nie w CSV)                                       |          |
| EDA z minimum 5 wizualizacjami                                  |          |
| Podział train/val/test                                          |          |
| Baseline model wytrenowany                                      |          |
| Metryki zapisane do JSON                                        |          |
| `.env.example` obecny, brak danych uwierzytelniających w kodzie |          |
| `.gitignore` poprawny                                           |          |

---

## Co dalej (Sprint 2)

Na następnym spotkaniu przenosimy kod z notebooka do **pipeline'u Kedro** — zautomatyzowanego przepływu danych od źródła do wyniku, uruchamialnego jednym poleceniem `kedro run`.
