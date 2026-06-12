import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

HOMEPLANET_MAP = {0: "Earth", 1: "Europa", 2: "Mars"}
DESTINATION_MAP = {0: "55 Cancri e", 1: "PSO J318.5-22", 2: "TRAPPIST-1e"}

st.set_page_config(
    page_title="🚀 Spaceship Titanic – Predykcja",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 Spaceship Titanic – Prediction Dashboard")

with st.sidebar:
    st.header("⚙️ Status API")

    if st.button("🔄 Sprawdź status", use_container_width=True):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            data = r.json()
            st.session_state["health"] = data
        except Exception as e:
            st.session_state["health"] = {"error": str(e)}

    health = st.session_state.get("health")
    if health and "error" not in health:
        st.success(f"Status: **{health['status']}**")
        st.info(f"Model: **{health['model_type']}**")
        loaded = "✅ Tak" if health["model_loaded"] else "❌ Nie"
        st.info(f"Załadowany: **{loaded}**")
    elif health and "error" in health:
        st.error(f"Błąd: {health['error']}")
    else:
        st.warning("Kliknij przycisk, aby sprawdzić status API")

    st.divider()
    st.subheader("📡 Endpointy")
    st.code("GET  /health", language="text")
    st.code("POST /predict", language="text")
    st.markdown(f"📖 [Dokumentacja Swagger]({API_URL}/docs)")

tab_predict, tab_batch, tab_explore = st.tabs([
    "🎯 Predykcja pojedyncza",
    "📊 Analiza wsadowa",
    "🔬 Eksploracja cech",
])

with tab_predict:
    st.subheader("Wprowadź cechy pasażera")

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

    total_spending = room_service + food_court + shopping_mall + spa + vr_deck

    spending_data = pd.DataFrame({
        "Kategoria": ["Room Service", "Food Court", "Shopping Mall", "Spa", "VR Deck"],
        "Kwota": [room_service, food_court, shopping_mall, spa, vr_deck],
    })

    if total_spending > 0:
        fig_spend = px.pie(
            spending_data, names="Kategoria", values="Kwota",
            title="Rozkład wydatków pasażera",
            color_discrete_sequence=px.colors.sequential.Purples_r,
            hole=0.45,
        )
        st.plotly_chart(fig_spend, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Planeta", HOMEPLANET_MAP[home_planet])
    m2.metric("Wiek", f"{age:.0f}")
    m3.metric("Kriosien", "Tak" if cryo_sleep else "Nie")
    m4.metric("Suma wydatków", f"${total_spending:,.0f}")

    st.divider()
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
                    model_name = result["model"]
                    transported = pred >= 0.5

                    if transported:
                        st.success("🌌 **TRANSPORTOWANY**")
                        st.balloons()
                    else:
                        st.error("🚫 **NIE TRANSPORTOWANY**")

                    # Store in history
                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].append({**payload, "prediction": pred, "model": model_name})
                else:
                    st.error(f"Błąd API ({r.status_code}): {r.text}")
            except requests.ConnectionError:
                st.error("❌ Nie udało się połączyć z API. Upewnij się, że serwer działa na `http://127.0.0.1:8000`")
            except Exception as e:
                st.error(f"Błąd: {e}")

with tab_batch:
    st.subheader("📊 Analiza wsadowa – historia predykcji")

    history = st.session_state.get("history", [])

    if not history:
        st.info("Brak danych. Wykonaj kilka predykcji w zakładce **Predykcja pojedyncza**, aby zobaczyć wyniki tutaj.")
    else:
        df_hist = pd.DataFrame(history)
        df_hist["Wynik"] = df_hist["prediction"].apply(lambda x: "Transportowany" if x >= 0.5 else "Nie transportowany")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Historia predykcji")
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

        with c2:
            counts = df_hist["Wynik"].value_counts().reset_index()
            counts.columns = ["Wynik", "Liczba"]
            fig_res = px.pie(counts, names="Wynik", values="Liczba",
                             color_discrete_sequence=["#667eea", "#f5576c"],
                             title="Rozkład wyników predykcji", hole=0.4)
            st.plotly_chart(fig_res, use_container_width=True)

        fig_dist = px.histogram(
            df_hist, x="prediction", nbins=20,
            title="Rozkład wartości predykcji",
            color_discrete_sequence=["#667eea"],
            labels={"prediction": "Wartość predykcji"},
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        if st.button("🗑️ Wyczyść historię"):
            st.session_state["history"] = []
            st.rerun()

with tab_explore:
    st.subheader("🔬 Eksploracja wpływu cechy na predykcję")
    st.caption("Wybierz jedną cechę do zbadania – pozostałe zostaną ustawione na wartości domyślne.")

    feature_to_explore = st.selectbox("Wybierz cechę", [
        "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "HomePlanet", "CryoSleep", "Destination", "VIP"
    ])

    defaults = {
        "HomePlanet": 0, "CryoSleep": 0, "Destination": 2,
        "Age": 30.0, "VIP": 0, "RoomService": 0.0,
        "FoodCourt": 0.0, "ShoppingMall": 0.0, "Spa": 0.0, "VRDeck": 0.0,
    }

    RANGES = {
        "Age": [float(x) for x in range(0, 81, 5)],
        "RoomService": [0, 50, 100, 200, 500, 1000, 2000, 5000],
        "FoodCourt": [0, 50, 100, 200, 500, 1000, 2000, 5000],
        "ShoppingMall": [0, 50, 100, 200, 500, 1000, 2000, 5000],
        "Spa": [0, 50, 100, 200, 500, 1000, 2000, 5000],
        "VRDeck": [0, 50, 100, 200, 500, 1000, 2000, 5000],
        "HomePlanet": [0, 1, 2],
        "CryoSleep": [0, 1],
        "Destination": [0, 1, 2],
        "VIP": [0, 1],
    }

    LABEL_MAPS = {
        "HomePlanet": HOMEPLANET_MAP,
        "CryoSleep": {0: "Nie", 1: "Tak"},
        "Destination": DESTINATION_MAP,
        "VIP": {0: "Nie", 1: "Tak"},
    }

    if st.button("📈 Uruchom eksplorację", type="primary", use_container_width=True):
        values = RANGES[feature_to_explore]
        results = []

        progress = st.progress(0, text="Odpytuję API...")
        for i, val in enumerate(values):
            payload = {**defaults, feature_to_explore: val}
            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if r.status_code == 200:
                    pred = r.json()["prediction"]
                else:
                    pred = None
            except Exception:
                pred = None
            results.append({"value": val, "prediction": pred})
            progress.progress((i + 1) / len(values), text=f"Przetwarzam {i+1}/{len(values)}...")

        progress.empty()
        df_explore = pd.DataFrame(results).dropna()

        if df_explore.empty:
            st.error("Nie udało się uzyskać predykcji. Sprawdź czy API działa.")
        else:
            if feature_to_explore in LABEL_MAPS:
                df_explore["label"] = df_explore["value"].map(LABEL_MAPS[feature_to_explore])
                x_col = "label"
            else:
                x_col = "value"

            is_categorical = feature_to_explore in LABEL_MAPS

            if is_categorical:
                fig = px.bar(
                    df_explore, x=x_col, y="prediction",
                    title=f"Wpływ cechy '{feature_to_explore}' na predykcję",
                    labels={x_col: feature_to_explore, "prediction": "Predykcja"},
                    color="prediction",
                    color_continuous_scale=["#f5576c", "#667eea"],
                )
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_explore["value"], y=df_explore["prediction"],
                    mode="lines+markers",
                    line=dict(color="#667eea", width=3),
                    marker=dict(size=8, color="#764ba2"),
                    fill="tozeroy",
                    fillcolor="rgba(102,126,234,0.1)",
                ))
                fig.update_layout(
                    title=f"Wpływ cechy '{feature_to_explore}' na predykcję",
                    xaxis_title=feature_to_explore,
                    yaxis_title="Predykcja",
                )

            fig.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Dane szczegółowe"):
                st.dataframe(df_explore, use_container_width=True, hide_index=True)

st.divider()
st.caption("Spaceship Titanic Prediction Dashboard · ASI Projekt · Sprint 6")
