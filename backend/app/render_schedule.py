from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt

DAY_ORDER = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab"]


def _day_index(name: str) -> int:
    try:
        return DAY_ORDER.index(name)
    except ValueError:
        return -1


def _abbr(text: str, max_len: int) -> str:
    text = str(text or "")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def render_schedule_png(schedule: Dict, cache_dir: str) -> bytes:
    classes: List[Dict] = schedule.get("classes", [])
    if not classes:
        return b""

    meetings: List[Dict] = []
    for cls in classes:
        raw_meetings = cls.get("meetings") or []
        if raw_meetings:
            for meeting in raw_meetings:
                meetings.append(
                    {
                        "day": meeting.get("day"),
                        "start": meeting.get("start_min", 0),
                        "end": meeting.get("end_min", 0),
                        "cls": cls,
                    }
                )
        else:
            start = cls.get("inicio", 0) or 0
            end = cls.get("fin", 0) or 0
            for day in cls.get("dias", []):
                meetings.append({"day": day, "start": start, "end": end, "cls": cls})

    start_hours = [(m.get("start", 0) or 0) / 60.0 for m in meetings]
    end_hours = [(m.get("end", 0) or 0) / 60.0 for m in meetings]

    min_h = max(7, int(min(start_hours) if start_hours else 7))
    max_h = min(22, int(max(end_hours) if end_hours else 22) + 1)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white", dpi=200)
    ax.set_xlim(-0.6, len(DAY_ORDER) - 0.4)
    ax.set_ylim(min_h, max_h)
    ax.set_xticks(range(len(DAY_ORDER)))
    ax.set_xticklabels(DAY_ORDER, fontsize=11, color="#111")
    ax.set_yticks(range(min_h, max_h + 1))
    ax.set_ylabel("Hora", fontsize=10, color="#444")
    ax.grid(color="#e5e7eb", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.set_facecolor("#ffffff")

    palette = ["#111111", "#2b2b2b", "#4a4a4a", "#6b6b6b"]
    for idx, meeting in enumerate(meetings):
        cls = meeting.get("cls", {})
        start = (meeting.get("start", 0) or 0) / 60.0
        end = (meeting.get("end", 0) or 0) / 60.0
        color = palette[idx % len(palette)]
        height = max(end - start, 0.25)

        dx = _day_index(meeting.get("day", ""))
        if dx < 0:
            continue
        rect = patches.FancyBboxPatch(
            (dx - 0.45, start),
            0.9,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#f8f8f8",
            edgecolor=color,
            linewidth=1.2,
        )
        ax.add_patch(rect)
        label = _abbr(cls.get("materia", ""), 18)
        time_txt = (
            f"{int(meeting.get('start',0)//60):02d}:{int(meeting.get('start',0)%60):02d}-"
            f"{int(meeting.get('end',0)//60):02d}:{int(meeting.get('end',0)%60):02d}"
        )
        ax.text(
            dx,
            start + height * 0.62,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            weight="bold",
        )
        ax.text(
            dx,
            start + height * 0.3,
            time_txt,
            ha="center",
            va="center",
            fontsize=8,
            color="#333",
        )

    title = f"Horario {schedule.get('schedule_id','')}"
    subtitle = "Escala basada en preferencias y reseñas"
    ax.set_title(title + "\n" + subtitle, fontsize=13, color="#0f172a", pad=16, loc="left")
    plt.tight_layout()

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    out_path = cache_path / f"{schedule.get('schedule_id','sched')}.png"
    plt.savefig(out_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(out_path, "rb") as f:
        return f.read()
