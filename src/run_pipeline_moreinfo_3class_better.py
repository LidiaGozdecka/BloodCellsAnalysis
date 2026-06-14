import os
import subprocess
import sys


def run_script(script_name):
    """Uruchamia podany skrypt Pythona i przekazuje jego wyjście na konsolę w czasie rzeczywistym."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n========================================================================")
    print(f"[PIPELINE 3-KLASY] URUCHAMIANIE: {script_name}")
    print(f"========================================================================\n")

    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1
    )

    collected_output = []

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
            collected_output.append(line)

    process.wait()

    if process.returncode != 0:
        print(f"\n[BŁĄD] Skrypt {script_name} zakończył się awarią!")
        sys.exit(1)

    print(f"\n[PIPELINE 3-KLASY] ZAKOŃCZONO SUKCESEM: {script_name}\n")
    return "".join(collected_output)


def generate_html_report(l3_output, l5_output):
    """Generuje raport w formacie HTML dostosowany do eksperymentu z 3 klasami."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.normpath(os.path.join(script_dir, ".."))
    report_path = os.path.join(project_dir, "raport_koncowy_3class_better.html")

    l3_plot_exists = os.path.exists(os.path.join(script_dir, "segmentacja_klasyczna_l3.png"))
    l5_plot_exists = os.path.exists(os.path.join(script_dir, "krzywe_uczenia_cnn_l5_3class.png"))

    l3_img_tag = '<div class="card"><img src="src/segmentacja_klasyczna_l3.png" alt="Wykres L3"></div>' if l3_plot_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku segmentacja_klasyczna_l3.png</p>'
    l5_img_tag = '<div class="card"><img src="src/krzywe_uczenia_cnn_l5_3class.png" alt="Krzywe uczenia L5 3-klasy"></div>' if l5_plot_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku krzywe_uczenia_cnn_l5_3class.png</p>'

    l3_cm_exists = os.path.exists(os.path.join(script_dir, "macierz_pomylek_l3.png"))
    l5_cm_exists = os.path.exists(os.path.join(script_dir, "macierz_pomylek_cnn_l5_3class.png"))

    l3_cm_tag = '<div class="card"><h4>Graficzna Macierz Pomyłek - L3</h4><img src="src/macierz_pomylek_l3.png" alt="Macierz L3"></div>' if l3_cm_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku macierz_pomylek_l3.png</p>'
    l5_cm_tag = '<div class="card"><h4>Graficzna Macierz Pomyłek - CNN (3 Klasy)</h4><img src="src/macierz_pomylek_cnn_l5_3class.png" alt="Macierz L5 3class"></div>' if l5_cm_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku macierz_pomylek_cnn_l5_3class.png</p>'

    html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Raport Końcowy (Zoptymalizowany) - Eksperyment 3-klasowy</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 40px; color: #333; background-color: #f4f6f9; }}
        h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #bdc3c7; padding-bottom: 10px; }}
        h1 {{ text-align: center; color: #16a085; margin-bottom: 30px; font-size: 2.5em; }}
        h2 {{ margin-top: 40px; color: #34495e; }}
        h4 {{ color: #7f8c8d; margin-bottom: 5px; }}
        pre {{ background-color: #1e272e; color: #d2dae2; padding: 20px; border-radius: 6px; overflow-x: auto; font-family: 'Consolas', monospace; font-size: 13px; border-left: 5px solid #16a085; line-height: 1.4; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
        .section {{ margin-bottom: 35px; }}
        .card {{ border: 1px solid #dcdde1; border-radius: 8px; padding: 15px; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.05); max-width: 600px; margin: 20px auto; text-align: center; }}
        .card img {{ width: 100%; height: auto; border-radius: 6px; display: block; margin-top: 10px; }}
        ul {{ padding-left: 20px; }}
        li {{ margin-bottom: 8px; }}
        .error-msg {{ color: #c0392b; font-style: italic; background: #fde8e7; padding: 10px; border-radius: 4px; border-left: 4px solid #e74c3c; margin: 10px 0; }}
        .highlight {{ background: #e8f8f5; border-radius: 4px; padding: 15px; border-left: 4px solid #16a085; margin: 15px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 25px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        th, td {{ border: 1px solid #dcdde1; padding: 14px; text-align: left; }}
        th {{ background-color: #2c3e50; color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #f1f2f6; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 Zintegrowany Raport Analityczny: Optymalizacja 3-Klasowa</h1>

    <div class="section">
        <h2>1. Opis i Zmiana Założeń Projektu (Biologiczna Konsolidacja Klas)</h2>
        <p>W pierwotnej wersji eksperymentu, z powodu skrajnego zaburzenia proporcji klas (dominacja neutrofili), sieć konwolucyjna wpadała w stan dezorientacji matematycznej. Wprowadzenie wag klasowych prowadziło do oscylacji wokół lokalnego minimum funkcji straty.</p>
        <div class="highlight">
            <strong>Modyfikacja Hematologiczna:</strong><br>
            W celu zniwelowania błędu, zastosowano fuzję popartą kryterium podziału biologicznego. Limfocyty i monocyty zostały połączone w jedną silniejszą klasę <strong>AGRANULOCYTÓW (komórek jednojądrzastych)</strong>. Klasa ta charakteryzuje się brakiem widocznych ziarnistości w cytoplazmie oraz zwartą budową jądra komórkowego, co ułatwia ekstrakcję spójnych cech przez warstwy splotowe.
        </div>
    </div>

    <div class="section">
        <h2>2. Podejście Klasyczne CV (L3) - Wyniki dla 4 Klas Bazowych</h2>
        <p>Punkt odniesienia dla danych wejściowych w celach porównawczych:</p>
        <pre>{l3_output}</pre>

        <h3>Wizualizacja Wyników Klasycznych:</h3>
        <div style="display: flex; gap: 20px; wrap: wrap; justify-content: center;">
            {l3_img_tag}
            {l3_cm_tag}
        </div>
    </div>

    <div class="section">
        <h2>3. Zoptymalizowana Sieć CNN (L5) - Wyniki dla 3 Klas po Fuzji</h2>
        <p>Architektura sieci splotowej trenowana na zbalansowanych wagach dla 3 skonsolidowanych grup diagnostycznych (Neutrofile, Eozynofile, Agranulocyty):</p>
        <pre>{l5_output}</pre>

        <h3>Wizualizacja Wyników Modelu 3-Klasowego:</h3>
        <div style="display: flex; gap: 20px; wrap: wrap; justify-content: center;">
            {l5_img_tag}
            {l5_cm_tag}
        </div>
    </div>

    <div class="section">
        <h2>4. Podsumowanie Wniosków</h2>
        <p>Konsolidacja struktur klasowych i sprowadzenie problemu do 3 klas pozwoliło sieci na stabilizację przebiegu funkcji straty (Loss). Szczegółowe zestawienie macierzy pomyłek jednoznacznie wskazuje, czy model przestał ślepo klasyfikować obrazy do jednej grupy dominującej.</p>
    </div>
</div>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[PIPELINE] SUKCES! Wygenerowano zoptymalizowany raport: {report_path}")


if __name__ == "__main__":
    print("=================== INTEGRACJA POTOKU: EKSPERYMENT 3-KLASOWY ===================")

    l3_logs = run_script("L3_pomiary.py")
    l5_logs = run_script("L5_cnn_3_class_better.py")

    generate_html_report(l3_logs, l5_logs)
    print("======================================================================================")


