import os
import sqlite3
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
DB_PATH = Path(os.environ.get("DB_PATH", "data/01_raw/dataset.db"))
DB_TABLE = os.environ.get("DB_TABLE", "spaceship_titanic")

SDV_DROP_COLUMNS = ["PassengerId", "Name", "Cabin"]

HOMEPLANET_MAP = {0: "Earth", 1: "Europa", 2: "Mars"}
DESTINATION_MAP = {0: "55 Cancri e", 1: "PSO J318.5-22", 2: "TRAPPIST-1e"}


@st.cache_data
def load_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(f"SELECT * FROM {DB_TABLE}", conn)


def _prepare_for_sdv(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [str(c) for c in clean.columns]
    drop = [c for c in SDV_DROP_COLUMNS if c in clean.columns]
    if drop:
        clean = clean.drop(columns=drop)
    for col in clean.columns:
        clean[col] = pd.to_numeric(clean[col], errors="ignore")
    return clean


@st.cache_resource
def fit_synthesizer(real_data: pd.DataFrame):
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer

    metadata = Metadata.detect_from_dataframe(real_data)
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(real_data)
    return synth


st.set_page_config(
    page_title="🚀 ASI Spaceship Titanic — Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 ASI Spaceship Titanic — Dashboard MLOps")

with st.sidebar:
    st.header("⚙️ Status API")
    st.caption(f"API_URL = `{API_URL}`")
    if st.button("🔄 Sprawdź status", use_container_width=True):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            st.session_state["health"] = r.json()
        except Exception as exc:
            st.session_state["health"] = {"error": str(exc)}

    health = st.session_state.get("health")
    if health and "error" not in health:
        st.success(f"Status: **{health.get('status')}**")
        st.info(f"Model: **{health.get('model_type')}**")
        st.info("Załadowany: " + ("✅ Tak" if health.get("model_loaded") else "❌ Nie"))
    elif health and "error" in health:
        st.error(f"Brak połączenia: {health['error']}")
    else:
        st.warning("Kliknij, aby sprawdzić status API.")

tab_app, tab_data, tab_synth = st.tabs([
    "🎯 Aplikacja",
    "🗄️ Dane z bazy",
    "🧪 Dane syntetyczne",
])

with tab_app:
    st.header("Aplikacja — predykcja transportu pasażera")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🌍 Dane osobowe**")
        home_planet = st.selectbox("Planeta pochodzenia", options=[0, 1, 2],
                                   format_func=lambda x: HOMEPLANET_MAP[x])
        destination = st.selectbox("Cel podróży", options=[0, 1, 2],
                                   format_func=lambda x: DESTINATION_MAP[x])
        age = st.slider("Wiek", 0.0, 120.0, 30.0, step=1.0)
    with col2:
        st.markdown("**🛏️ Status**")
        cryo_sleep = st.selectbox("Kriogeniczny sen", options=[0, 1],
                                  format_func=lambda x: "Tak" if x else "Nie")
        vip = st.selectbox("VIP", options=[0, 1],
                           format_func=lambda x: "Tak" if x else "Nie")
    with col3:
        st.markdown("**💰 Wydatki na pokładzie**")
        room_service = st.number_input("Room Service", min_value=0.0, value=0.0, step=50.0)
        food_court = st.number_input("Food Court", min_value=0.0, value=0.0, step=50.0)
        shopping_mall = st.number_input("Shopping Mall", min_value=0.0, value=0.0, step=50.0)
        spa = st.number_input("Spa", min_value=0.0, value=0.0, step=50.0)
        vr_deck = st.number_input("VR Deck", min_value=0.0, value=0.0, step=50.0)

    if st.button("🚀 Wykonaj predykcję", type="primary", use_container_width=True):
        payload = {
            "HomePlanet": home_planet,
            "CryoSleep": cryo_sleep,
            "Destination": destination,
            "Age": age,
            "VIP": vip,
            "RoomService": room_service,
            "FoodCourt": food_court,
            "ShoppingMall": shopping_mall,
            "Spa": spa,
            "VRDeck": vr_deck,
        }
        with st.spinner("Wysyłam zapytanie do API..."):
            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if r.status_code == 200:
                    result = r.json()
                    pred = result["prediction"]
                    transported = pred >= 0.5
                    if transported:
                        st.success(f"🌌 **TRANSPORTOWANY** (p = {pred:.3f})")
                        st.balloons()
                    else:
                        st.error(f"🚫 **NIE TRANSPORTOWANY** (p = {pred:.3f})")
                    st.caption(f"Model: {result.get('model')}")
                elif r.status_code == 422:
                    st.error("Błędne dane wejściowe (walidacja Pydantic).")
                    st.json(r.json())
                elif r.status_code == 503:
                    st.error("API działa, ale model nie jest załadowany. "
                             "Uruchom `kedro run` i zrestartuj API.")
                else:
                    st.error(f"Błąd API ({r.status_code}): {r.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Nie można połączyć się z API pod {API_URL}. "
                         "Czy `uvicorn api.main:app` jest uruchomione?")
            except requests.exceptions.Timeout:
                st.error("⏱️ Przekroczono czas oczekiwania na odpowiedź API.")

with tab_data:
    st.header("Dane z bazy (SQLite)")
    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Nie udało się wczytać danych z bazy `{DB_PATH}`: {exc}")
        df = None

    if df is not None:
        st.write(f"Liczba rekordów: **{len(df)}** · liczba kolumn: **{df.shape[1]}**")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Statystyki opisowe")
        st.dataframe(df.describe(include="all"), use_container_width=True)

        st.subheader("Rozkład wybranej kolumny")
        numeric_cols = list(df.select_dtypes("number").columns)
        if numeric_cols:
            column = st.selectbox("Kolumna", numeric_cols)
            st.bar_chart(df[column].value_counts().sort_index())
        else:
            st.info("Brak kolumn numerycznych do wizualizacji.")

with tab_synth:
    st.header("Dane syntetyczne (SDV)")
    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Nie udało się wczytać danych z bazy: {exc}")
        df = None

    if df is not None:
        real_for_sdv = _prepare_for_sdv(df)
        st.caption("Kolumny użyte do generowania: " + ", ".join(real_for_sdv.columns))

        n_samples = st.number_input(
            "Liczba rekordów do wygenerowania", 100, 10_000, 1000, step=100
        )

        if st.button("Generuj dane syntetyczne", type="primary"):
            with st.spinner("Trenowanie synthesizera i generowanie..."):
                synth = fit_synthesizer(real_for_sdv)  # cache'owane — fit tylko raz
                st.session_state["synthetic"] = synth.sample(num_rows=int(n_samples))

        if "synthetic" in st.session_state:  # przetrwało rerun
            synthetic = st.session_state["synthetic"]
            st.success(f"Wygenerowano {len(synthetic)} rekordów.")

            col_real, col_synth = st.columns(2)
            with col_real:
                st.subheader("Oryginał (statystyki)")
                st.dataframe(real_for_sdv.describe(), use_container_width=True)
            with col_synth:
                st.subheader("Syntetyczne (statystyki)")
                st.dataframe(synthetic.describe(), use_container_width=True)

            st.subheader("Podgląd danych syntetycznych")
            st.dataframe(synthetic.head(100), use_container_width=True)

            st.download_button(
                "⬇️ Pobierz dane syntetyczne (CSV)",
                synthetic.to_csv(index=False).encode("utf-8"),
                file_name="synthetic_data.csv",
                mime="text/csv",
            )

st.divider()
st.caption("ASI Spaceship Titanic · Sprint 6 · Streamlit + FastAPI + SDV")
