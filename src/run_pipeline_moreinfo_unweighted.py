import os
import subprocess
import sys


def run_script(script_name):
    """Uruchamia podany skrypt Pythona i przekazuje jego wyjście na konsolę w czasie rzeczywistym."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n========================================================================")
    print(f"[PIPELINE BAZOWY] URUCHAMIANIE: {script_name}")
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

    print(f"\n[PIPELINE BAZOWY] ZAKOŃCZONO SUKCESEM: {script_name}\n")
    return "".join(collected_output)


def generate_html_report(l3_output, l5_output):
    """Generuje raport HTML obrazujący problem ignorowania mniejszościowych klas."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.normpath(os.path.join(script_dir, ".."))
    report_path = os.path.join(project_dir, "raport_bazowy_unweighted.html")

    l3_plot_exists = os.path.exists(os.path.join(script_dir, "segmentacja_klasyczna_l3.png"))
    l5_plot_exists = os.path.exists(os.path.join(script_dir, "krzywe_uczenia_cnn_l5_unweighted.png"))

    l3_img_tag = '<div class="card"><img src="src/segmentacja_klasyczna_l3.png" alt="Wykres L3"></div>' if l3_plot_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku segmentacja_klasyczna_l3.png</p>'
    l5_img_tag = '<div class="card"><img src="src/krzywe_uczenia_cnn_l5_unweighted.png" alt="Krzywe uczenia L5 bazowe"></div>' if l5_plot_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku krzywe_uczenia_cnn_l5_unweighted.png</p>'

    l3_cm_exists = os.path.exists(os.path.join(script_dir, "macierz_pomylek_l3.png"))
    l5_cm_exists = os.path.exists(os.path.join(script_dir, "macierz_pomylek_cnn_l5_unweighted.png"))

    l3_cm_tag = '<div class="card"><h4>Graficzna Macierz Pomyłek - L3</h4><img src="src/macierz_pomylek_l3.png" alt="Macierz L3"></div>' if l3_cm_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku macierz_pomylek_l3.png</p>'
    l5_cm_tag = '<div class="card"><h4>Graficzna Macierz Pomyłek - CNN (Bez wag)</h4><img src="src/macierz_pomylek_cnn_l5_unweighted.png" alt="Macierz L5 unweighted"></div>' if l5_cm_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku macierz_pomylek_cnn_l5_unweighted.png</p>'

    html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Raport Pierwotny - Problem Niezbalansowania</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 40px; color: #333; background-color: #f4f6f9; }}
        h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #bdc3c7; padding-bottom: 10px; }}
        h1 {{ text-align: center; color: #c0392b; margin-bottom: 30px; font-size: 2.5em; }}
        h2 {{ margin-top: 40px; color: #34495e; }}
        h4 {{ color: #7f8c8d; margin-bottom: 5px; }}
        pre {{ background-color: #1e272e; color: #d2dae2; padding: 20px; border-radius: 6px; overflow-x: auto; font-family: 'Consolas', monospace; font-size: 13px; border-left: 5px solid #c0392b; line-height: 1.4; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
        .section {{ margin-bottom: 35px; }}
        .card {{ border: 1px solid #dcdde1; border-radius: 8px; padding: 15px; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.05); max-width: 600px; margin: 20px auto; text-align: center; }}
        .card img {{ width: 100%; height: auto; border-radius: 6px; display: block; margin-top: 10px; }}
        .warning-highlight {{ background: #fdf2e9; border-radius: 4px; padding: 15px; border-left: 4px solid #e67e22; margin: 15px 0; color: #d35400; font-weight: 500; }}
        .error-msg {{ color: #c0392b; font-style: italic; background: #fde8e7; padding: 10px; border-radius: 4px; border-left: 4px solid #e74c3c; margin: 10px 0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🚨 Pierwotny Raport Badawczy: Problem Dominacji Klasowej CNN</h1>

    <div class="section">
        <h2>1. Założenia Eksperymentu Bazowego</h2>
        <p>Niniejszy dokument przedstawia wyjściowe (naiwne) podejście do uczenia sieci CNN bez implementacji algorytmów równoważenia zbioru danych. Celowo zignorowano asymetrię liczebnościową próbek, by zweryfikować zachowanie standardowej funkcji straty <code>categorical_crossentropy</code> na rzecz klasy dominującej.</p>
        <div class="warning-highlight">
            ⚠️ OSTRZEŻENIE DIAGNOSTYCZNE:<br>
            W tym podejściu sieć osiąga pozornie wysoką celność (Accuracy ok. 60%), która wynika wyłącznie z faktu, że klasa NEUTROPHIL stanowi większość zbioru. Model wykazuje lenistwo matematyczne i ignoruje rzadkie klasy komórek.
        </div>
    </div>

    <div class="section">
        <h2>2. Podejście Klasyczne CV (L3) - Stan Odniesienia</h2>
        <pre>{l3_output}</pre>
        <h3>Wizualizacja Wyników Klasycznych:</h3>
        <div style="display: flex; gap: 20px; wrap: wrap; justify-content: center;">
            {l3_img_tag}
            {l3_cm_tag}
        </div>
    </div>

    <div class="section">
        <h2>3. Naiwna Sieć CNN (L5_unweighted) - Wyniki i Patologie Uczenia</h2>
        <p>Poniższe logi oraz macierz pomyłek obrazują brak zdolności generalizacji sieci dla klas: Eosinophil, Lymphocyte oraz Monocyte.</p>
        <pre>{l5_output}</pre>

        <h3>Wizualizacja Patologii Uczenia (Płaskie linie celności i zablokowana strata):</h3>
        <div style="display: flex; gap: 20px; wrap: wrap; justify-content: center;">
            {l5_img_tag}
            {l5_cm_tag}
        </div>
    </div>

    <div class="section">
        <h2>4. Krytyczny Wniosek Badawczy</h2>
        <p>Przedstawiony powyżej wykres predykcji (macierz pomyłek w formie pojedynczej, pionowej kolumny) udowadnia, że bez interwencji w postaci wag klasowych lub modyfikacji struktury etykiet, model staje się bezużyteczny z punktu widzenia diagnostyki medycznej, mimo "ładnego" wykresu celności ogólnej.</p>
    </div>
</div>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[PIPELINE] SUKCES! Wygenerowano raport wadliwej wersji: {report_path}")


if __name__ == "__main__":
    print("=================== INTEGRACJA POTOKU: MODEL NIEZBALANSOWANY ===================")

    l3_logs = run_script("L3_pomiary_better.py")
    l5_logs = run_script("L5_cnn_unweighted.py")

    generate_html_report(l3_logs, l5_logs)
    print("======================================================================================")