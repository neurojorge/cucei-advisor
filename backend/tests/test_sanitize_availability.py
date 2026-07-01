import pandas as pd

from app.data_access import sanitize_availability


def test_sanitize_all_zero_to_none():
    df = pd.DataFrame({"cupo": [0, 0, 0], "disponibles": [0, 0, 0]})
    out, meta = sanitize_availability(df)
    assert meta["offer_avail_all_zero"] is True
    assert out["cupo"].isna().all()
    assert out["disponibles"].isna().all()


def test_sanitize_preserves_zero_when_some_available():
    df = pd.DataFrame({"cupo": [0, 20, 30], "disponibles": [0, 5, 0]})
    out, meta = sanitize_availability(df)
    assert meta["offer_avail_all_zero"] is False
    assert out.loc[0, "disponibles"] == 0
    assert out.loc[0, "cupo"] == 0


def test_sanitize_all_zero_with_nan_to_none():
    df = pd.DataFrame({"cupo": [0, None], "disponibles": [0, None]})
    out, meta = sanitize_availability(df)
    assert meta["offer_avail_all_zero"] is True
    assert out["cupo"].isna().all()
    assert out["disponibles"].isna().all()
