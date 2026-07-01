from app.groq_client import build_summary_generated


def test_summary_generated_fallback_without_reviews(tmp_path):
    summary = build_summary_generated(
        [],
        {"total": 0, "comments": []},
        {"barco": 0, "exigente": 0},
        cache_key="profesor-demo",
        cache_dir=tmp_path.as_posix(),
    )
    assert summary["summary_text"].lower().startswith("sin")
    assert summary["confidence"] == 0
    assert summary["tags"] == {"barco": 0, "exigente": 0}
