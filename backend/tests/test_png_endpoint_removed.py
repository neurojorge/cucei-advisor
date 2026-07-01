from app.main import app


def test_png_route_removed_from_openapi():
    schema = app.openapi()
    paths = schema.get("paths", {})
    assert "/api/schedule/{schedule_id}/png" not in paths
