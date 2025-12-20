import pandas as pd

from app.adapters import adapt_schema_offer


def test_adapt_schema_offer_maps_carrera():
    df = pd.DataFrame(
        {
            "carrera": ["Informatica", "Computacion"],
            "clave": ["I1234", "I5678"],
            "materia": ["Algoritmos", "Redes"],
            "nrc": ["1001", "1002"],
            "seccion": ["A", "B"],
            "profesor": ["Prof A", "Prof B"],
            "horario": ["0700-0855 L", "0900-1055 M"],
            "cupo": [20, 30],
            "disponibles": [5, 0],
        }
    )
    out = adapt_schema_offer(df)
    for col in [
        "carrera_clave",
        "clave_materia",
        "materia",
        "nrc",
        "seccion",
        "profesor",
        "horario_raw",
    ]:
        assert col in out.columns
    assert out.loc[0, "carrera_clave"] == "INFO"
    assert out.loc[1, "carrera_clave"] == "ICOM"
