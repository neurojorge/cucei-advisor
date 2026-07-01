from pathlib import Path

from app.plan_loader import get_semester_keys, load_plan_df


def test_plan_core_filtering():
    plan_path = Path(__file__).resolve().parents[2] / "semestres_materias_INFO_ICOM.csv"
    df = load_plan_df(plan_path.as_posix())
    keys = get_semester_keys(df, "INFO", 3, group="CORE")
    assert keys, "Expected CORE keys for INFO semestre 3"
    subset = df[(df["carrera_clave"] == "INFO") & (df["semestre"] == 3) & (df["grupo"] == "CORE")]
    assert set(keys).issubset(set(subset["clave_materia"].unique().tolist()))
