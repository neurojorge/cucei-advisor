from app.scheduler import build_meetings, parse_timeslots, schedule_has_conflicts, sections_conflict, Section


def _make_section(name: str, row: dict) -> Section:
    slots = parse_timeslots(row)
    return Section(
        materia=name,
        clave=name,
        nrc="1",
        seccion="A",
        profesor="Prof",
        cupo=20,
        disponibles=5,
        carrera="INFO",
        semestre=1,
        slots=slots,
        raw=row.get("horario_raw", ""),
    )


def test_parse_and_conflict_detection():
    row_a = {
        "dias": "L . . . . .",
        "hora_inicio": "0900",
        "hora_fin": "1055",
        "horario_raw": "0900-1055 L",
    }
    row_b = {
        "dias": "L . . . . .",
        "hora_inicio": "1000",
        "hora_fin": "1155",
        "horario_raw": "1000-1155 L",
    }
    row_c = {
        "dias": "M . . . . .",
        "hora_inicio": "1200",
        "hora_fin": "1355",
        "horario_raw": "1200-1355 M",
    }

    sec_a = _make_section("A", row_a)
    sec_b = _make_section("B", row_b)
    sec_c = _make_section("C", row_c)

    assert sections_conflict(sec_a, sec_b) is True
    assert sections_conflict(sec_a, sec_c) is False


def test_schedule_has_conflicts_and_meetings():
    row_a = {
        "dias": "L . . . . .",
        "hora_inicio": "0900",
        "hora_fin": "1055",
        "horario_raw": "0900-1055 L",
    }
    row_b = {
        "dias": "M . . . . .",
        "hora_inicio": "1200",
        "hora_fin": "1355",
        "horario_raw": "1200-1355 M",
    }
    sec_a = _make_section("A", row_a)
    sec_b = _make_section("B", row_b)
    assert schedule_has_conflicts([sec_a, sec_b]) is False
    meetings = build_meetings(sec_a.slots)
    assert meetings == [{"day": "Lun", "start_min": 540, "end_min": 655}]
