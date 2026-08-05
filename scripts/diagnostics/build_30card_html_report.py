"""31.07 (cd. 8) — Łączy liczbową tabelę z 30-kartowego sweepu (`all_methods_30card_sweep.py`)
z gotowymi wizualizacjami z `all_methods_comparison_report.py` (v8) w jeden raport HTML.

Nie przelicza NICZEGO od nowa — czysta sklejka dwóch już istniejących artefaktów:
  - `outputs/31.07_ring_shortest_path/30card_sweep_all_methods.json` (tabela, 30 kart)
  - `outputs/31.07_ring_shortest_path/report_v8_all_methods.html` (obrazy, 2 przykładowe otolity)

WAŻNE zastrzeżenie (wprost w raporcie): obrazy pochodzą z INNEGO, mniejszego zestawienia (2
otolity, niekoniecznie te same co w 30-kartowej próbie) — są ilustracją MECHANIZMU każdej
metody, NIE dowodem liczb w tabeli. Tabela i obrazy są więc pokazane razem, ale jawnie opisane
jako dwa osobne źródła.

Usage: PYTHONIOENCODING=utf-8 python scripts/diagnostics/build_30card_html_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
OUT_DIR = PROJECT_ROOT / "outputs" / "31.07_ring_shortest_path"

JSON_PATH = OUT_DIR / "30card_sweep_all_methods.json"
V8_HTML_PATH = OUT_DIR / "report_v8_all_methods.html"
OUT_PATH = OUT_DIR / "30card_sweep_all_methods.html"

# Mapowanie: nazwa metody w tabeli -> fragment nagłówka <h2> w report_v8_all_methods.html
# (do wyciągnięcia właściwej sekcji z obrazami). Metody produkcyjne (classical/dp/consensus/
# density) nie mają odpowiednika w v8 (nie są tam wizualizowane) — brak obrazu, tylko liczba.
METHOD_TO_V8_HEADER = {
    "gradient — closed shortest-path": "3. Circular shortest path",
    "Frangi — closed shortest-path": "3. Circular shortest path",
    "gradient — pewne otwarte fragmenty": "4. Pewne otwarte fragmenty",
    "Frangi — pewne otwarte fragmenty": "4. Pewne otwarte fragmenty",
    "Gabor — pewne otwarte fragmenty": "7. Bank filtrów Gabora",
    "CLAHE+density+Frangi — pewne otwarte fragmenty": "6. CLAHE",
    "Frangi — RANSAC": "5. RANSAC",
    "Hough": "8. Transformata Hougha",
    "snake (t0=0,5)": "9. Aktywne kontury",
}


def extract_sections(html: str) -> dict[str, str]:
    """Zwraca {początek nagłówka h2 -> pełny blok <section>...</section>}."""
    sections = re.findall(r"<section>.*?</section>", html, flags=re.S)
    out = {}
    for sec in sections:
        m = re.search(r"<h2>(.*?)</h2>", sec, flags=re.S)
        if m:
            out[m.group(1).strip()] = sec
    return out


def find_section_for(sections: dict[str, str], needle: str) -> str | None:
    for header, sec in sections.items():
        if needle in header:
            return sec
    return None


def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    summary = data["summary"]

    v8_html = V8_HTML_PATH.read_text(encoding="utf-8")
    v8_sections = extract_sections(v8_html)

    rows_html = []
    used_headers: set[str] = set()
    for name, n, val in summary:
        val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
        star = " *" if n == 30 and any(name.startswith(p) for p in
                                       ("classical (produkcyjny", "consensus (produkcyjny",
                                        "dp (produkcyjny", "density (produkcyjny")) else ""
        rows_html.append(f"<tr><td>{name}{star}</td><td>{n}</td><td><b>{val_str}</b></td></tr>")
    table_html = f"""<table>
<tr><th>Metoda</th><th>n kart</th><th>mean_dist (px)</th></tr>
{"".join(rows_html)}
</table>
<p style="font-size:85%;color:#555;">(*) cytowane z <code>outputs/28.07_e9_w0.1/
localization_quality.json</code>, ta sama próba, NIE przeliczane od nowa.</p>"""

    image_sections = []
    for name, needle in METHOD_TO_V8_HEADER.items():
        sec = find_section_for(v8_sections, needle)
        if sec and needle not in used_headers:
            image_sections.append(sec)
            used_headers.add(needle)

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>30-kartowe porównanie wszystkich metod — tabela + obrazy</title>
<style>
body {{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
h1 {{color:#1a237e;}} h2 {{color:#1a237e;font-size:110%;}}
code {{background:#f0f0f5;padding:1px 4px;border-radius:3px;}}
table {{border-collapse:collapse;margin:1em 0;}}
td,th {{padding:5px 12px;border:1px solid #ddd;font-size:92%;text-align:left;}}
th {{background:#f0f0f5;}}
tr:nth-child(even) {{background:#fafafa;}}
</style>
</head>
<body>
<h1>Liczbowe porównanie wszystkich metod — 30 kart + wizualizacje</h1>
<p>Tabela poniżej to wynik <code>scripts/diagnostics/all_methods_30card_sweep.py</code> — TA SAMA
30-kartowa próba (15 najlepszych + 15 najgorszych wg błędu wieku) co cała historia projektu od
21.07. Metody produkcyjne (classical/dp/consensus/density) cytowane wprost, nie przeliczane.</p>
<p style="background:#fff3cd;padding:8px;border-radius:4px;font-size:90%;"><b>Ważne zastrzeżenie:</b>
obrazy PONIŻEJ tabeli pochodzą z OSOBNEGO, mniejszego zestawienia
(<code>outputs/31.07_ring_shortest_path/report_v8_all_methods.html</code>, 2 przykładowe otolity,
niekoniecznie te same co w 30-kartowej próbie) — to ILUSTRACJA MECHANIZMU każdej metody, nie dowód
liczb w tabeli. Traktuj tabelę i obrazy jako dwa osobne, uzupełniające się źródła.</p>
<h2 style="border-top:none;">Tabela — mean_dist_final_to_classical_px (px, niżej=lepiej)</h2>
{table_html}
<h1 style="margin-top:2em;">Wizualizacje mechanizmów (2 przykładowe otolity, z v8)</h1>
{"".join(image_sections)}
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Zapisano: {OUT_PATH}")


if __name__ == "__main__":
    main()
