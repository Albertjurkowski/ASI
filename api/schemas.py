from pydantic import BaseModel, Field


class SpaceshipFeatures(BaseModel):
    """Cechy pasażera po preprocessingu (LabelEncoded + numeryczne).

    Wszystkie pola odpowiadają kolumnom jakie model widział podczas treningu —
    po zakodowaniu kategorycznych przez LabelEncoder i uzupełnieniu braków medianą.
    """

    HomePlanet: int = Field(..., ge=0, le=10, description="LabelEncoded planeta pochodzenia (Earth/Europa/Mars)")
    CryoSleep: int = Field(..., ge=0, le=1, description="Kriogeniczny sen: 0=nie, 1=tak")
    Destination: int = Field(..., ge=0, le=10, description="LabelEncoded cel podróży")
    Age: float = Field(..., ge=0.0, le=120.0, description="Wiek pasażera")
    VIP: int = Field(..., ge=0, le=1, description="Status VIP: 0=nie, 1=tak")
    RoomService: float = Field(..., ge=0.0, description="Wydatki na obsługę pokojową")
    FoodCourt: float = Field(..., ge=0.0, description="Wydatki w food court")
    ShoppingMall: float = Field(..., ge=0.0, description="Wydatki w centrum handlowym")
    Spa: float = Field(..., ge=0.0, description="Wydatki w SPA")
    VRDeck: float = Field(..., ge=0.0, description="Wydatki na VR Deck")

    model_config = {
        "json_schema_extra": {
            "example": {
                "HomePlanet": 1,
                "CryoSleep": 0,
                "Destination": 2,
                "Age": 35.0,
                "VIP": 0,
                "RoomService": 0.0,
                "FoodCourt": 500.0,
                "ShoppingMall": 200.0,
                "Spa": 0.0,
                "VRDeck": 0.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predykcja modelu (0=nie transportowany, 1=transportowany)")
    model: str = Field(..., description="Nazwa użytego modelu")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
