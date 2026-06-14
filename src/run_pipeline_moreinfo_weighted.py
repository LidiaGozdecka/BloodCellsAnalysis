import os
import subprocess
import sys


def run_script(script_name):
    """Uruchamia podany skrypt Pythona i przekazuje jego wyjście na konsolę w czasie rzeczywistym."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n========================================================================")
    print(f"[PIPELINE] URUCHAMIANIE: {script_name}")
    print(f"========================================================================\n")

    # Uruchamiamy proces z przekierowaniem strumieni wyjściowych
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1  # Wymuszenie buforowania linijka po linijce
    )

    collected_output = []

    # Czytamy wyjście na żywo, linijka po linijce
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            sys.stdout.write(line)  # Wypisujemy od razu na ekran użytkownika
            sys.stdout.flush()
            collected_output.append(line)

    process.wait()

    if process.returncode != 0:
        print(f"\n[BŁĄD] Skrypt {script_name} zakończył się awarią (Exit code: {process.returncode})!")
        sys.exit(1)

    print(f"\n[PIPELINE] ZAKOŃCZONO SUKCESEM: {script_name}\n")
    return "".join(collected_output)


def generate_html_report(l3_output, l5_output):
    """Generuje ostateczny raport w formacie HTML na podstawie zebranych danych."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.normpath(os.path.join(script_dir, ".."))
    report_path = os.path.join(project_dir, "raport_koncowy_weighted.html")

    # 1. Sprawdzanie podstawowych wykresów (Krzywe uczenia i segmentacja)
    l3_plot_exists = os.path.exists(os.path.join(script_dir, "segmentacja_klasyczna_l3.png"))
    l5_plot_exists = os.path.exists(os.path.join(script_dir, "krzywe_uczenia_cnn_l5.png"))

    l3_img_tag = '<div class="card"><img src="src/segmentacja_klasyczna_l3.png" alt="Wykres L3"></div>' if l3_plot_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku segmentacja_klasyczna_l3.png</p>'
    l5_img_tag = '<div class="card"><img src="src/krzywe_uczenia_cnn_l5.png" alt="Krzywe uczenia L5"></div>' if l5_plot_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku krzywe_uczenia_cnn_l5.png</p>'

    # 2. Sprawdzanie graficznych macierzy pomyłek (Seaborn Heatmaps) - POPRAWIONA LOKALIZACJA
    l3_cm_exists = os.path.exists(os.path.join(script_dir, "macierz_pomylek_l3.png"))
    l5_cm_exists = os.path.exists(os.path.join(script_dir, "macierz_pomylek_cnn_l5.png"))

    l3_cm_tag = '<div class="card"><h4>Graficzna Macierz Pomyłek - L3</h4><img src="src/macierz_pomylek_l3.png" alt="Macierz L3"></div>' if l3_cm_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku macierz_pomylek_l3.png</p>'
    l5_cm_tag = '<div class="card"><h4>Graficzna Macierz Pomyłek - L5</h4><img src="src/macierz_pomylek_cnn_l5.png" alt="Macierz L5"></div>' if l5_cm_exists else '<p class="error-msg">⚠️ Nie znaleziono pliku macierz_pomylek_cnn_l5.png</p>'

    html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Raport Końcowy - Analiza i Klasyfikacja Obrazów Krwinek</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 40px; color: #333; background-color: #f4f6f9; }}
        h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #bdc3c7; padding-bottom: 10px; }}
        h1 {{ text-align: center; color: #2980b9; margin-bottom: 30px; font-size: 2.5em; }}
        h2 {{ margin-top: 40px; color: #34495e; }}
        h4 {{ color: #7f8c8d; margin-bottom: 5px; }}
        pre {{ background-color: #1e272e; color: #d2dae2; padding: 20px; border-radius: 6px; overflow-x: auto; font-family: 'Consolas', monospace; font-size: 13px; border-left: 5px solid #2980b9; line-height: 1.4; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
        .section {{ margin-bottom: 35px; }}
        .card {{ border: 1px solid #dcdde1; border-radius: 8px; padding: 15px; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.05); max-width: 600px; margin: 20px auto; text-align: center; }}
        .card img {{ width: 100%; height: auto; border-radius: 6px; display: block; margin-top: 10px; }}
        ul {{ padding-left: 20px; }}
        li {{ margin-bottom: 8px; }}
        .error-msg {{ color: #c0392b; font-style: italic; background: #fde8e7; padding: 10px; border-radius: 4px; border-left: 4px solid #e74c3c; margin: 10px 0; }}
        .highlight {{ background: #f1f2f6; border-radius: 4px; padding: 15px; border-left: 4px solid #718093; margin: 15px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 25px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        th, td {{ border: 1px solid #dcdde1; padding: 14px; text-align: left; }}
        th {{ background-color: #2c3e50; color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #f1f2f6; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 Zintegrowany Raport Analityczny: Klasyfikacja Leukocytów</h1>

    <div class="section">
        <h2>1. Opis Zbioru Danych i Problem Niezbalansowania Klas</h2>
        <p>Projekt realizowany jest na bazie danych mikroskopowych zawierających obrazy białych krwinek (WBC). Analiza wstępna pliku <code>labels.csv</code> wykazała silną asymetrię liczebnościową klas (Neutrofile: 217, Eozynofile: 91, Limfocyty: 34, Monocyty: 22). Ze względu na krytycznie niską reprezentację klasy <strong>BASOPHIL</strong> (jedynie 3 wystąpienia w całym zbiorze), została ona wyłączona z procesu modelowania w celu uniknięcia błędów stratyfikacji danych (<code>train_test_split</code>) podczas podziału na zbiory uczący, walidacyjny i testowy.</p>
    </div>

    <div class="section">
        <h2>2. Podejście Klasyczne CV (L3) - Segmentacja i Reguły Geometryczne</h2>
        <h3>Zasady i Reguły Klasyfikacji dla Typów Komórek:</h3>
        <ul>
            <li><strong>EOSINOPHIL:</strong> Detekcja obecności jądra w masce fioletowej (HSV) skorelowana z silną obecnością ziarnistości kwasochłonnych w dedykowanej masce czerwonej (wyznaczonej z połączonych krańców osi Hue).</li>
            <li><strong>NEUTROPHIL:</strong> Komórki charakteryzujące się jądrem wielopłatowym (liczba segmentów wyznaczona transformatą dystansową i Connective Components > 1) lub niskim współczynnikiem okrągłości (&lt; 0.55).</li>
            <li><strong>LYMPHOCYTE:</strong> Małe jądra o regularnym kształcie, charakteryzujące się wysoką okrągłością (&ge; 0.75) oraz małą powierzchnią całkowitą (&lt; 7500 px).</li>
            <li><strong>MONOCYTE:</strong> Duże komórki o pofalowanym, nieregularnymjądre jednopłatowym, niespełniające restrykcyjnych kryteriów geometrycznych limfocytów.</li>
        </ul>

        <h3>Wyniki Ekstrakcji i Klasyfikacji Regułowej (Konsola + Macierz Pomyłek):</h3>
        <pre>{l3_output}</pre>

        <h3>Wizualizacja Wyników Klasycznych:</h3>
        <div style="display: flex; gap: 20px; wrap: wrap; justify-content: center;">
            {l3_img_tag}
            {l3_cm_tag}
        </div>
    </div>

    <div class="section">
        <h2>3. Podejście Głębokie - Sieć Konwolucyjna CNN (L5)</h2>
        <h3>Opis Parametrów i Architektury Sieci:</h3>
        <ul>
            <li><strong>Warstwa Wejściowa:</strong> Skalowanie obrazu wejściowego do rozdzielczości 128x128 pikseli o 3 kanałach barwnych (RGB).</li>
            <li><strong>Segment Ekstrakcji Cech (Splotowy):</strong> 3 sekwencyjne bloki konwolucyjne (Conv2D: 32, 64, 128 filtrów z funkcją aktywacji ReLU) połączone bezpośrednio z warstwami redukcji przestrzennej MaxPooling2D (rozmiar okna 2x2).</li>
            <li><strong>Klasyfikator Gęsty:</strong> Warstwa spłaszczająca Flatten, warstwa gęsta Dense (128 neuronów), warstwa regularyzacyjna <strong>Dropout(0.5)</strong> redukująca ryzyko przeuczenia, oraz wyjściowa warstwa Softmax z 4 neuronami odpowiadającymi klasom decyzyznym.</li>
            <li><strong>Optymalizator i Hiperparametry:</strong> Algorytm Adam, funkcja straty: <code>categorical_crossentropy</code>. Trening przeprowadzony na przestrzeni 50 epok przy wielkości paczki (Batch Size) równej 64.</li>
        </ul>

        <div class="highlight">
            <strong>Uruchomiona Augmentacja Danych (Wymogi z opisu PDF):</strong><br>
            Ze względu na małą liczebność zbioru bazowego (242 obrazy w zbiorze uczącym), wdrożono transformacje w locie (on-the-fly) za pomocą narzędzia <code>ImageDataGenerator</code>. Zastosowano: rotacje do 45°, losowe przesunięcia osiowe o 20%, losowy zoom o 15% oraz <strong>losową modyfikację jasności w przedziale [0.7, 1.3]</strong>. Łącznie przez sieć przeszło 12 100 unikalnych transformacji obrazu.
        </div>

        <h3>Wyniki Procesu Uczenia, Ewaluacji i Macierz Pomyłek CNN:</h3>
        <pre>{l5_output}</pre>

        <h3>Wizualizacja Wyników Modelu Głębokiego:</h3>
        <div style="display: flex; gap: 20px; wrap: wrap; justify-content: center;">
            {l5_img_tag}
            {l5_cm_tag}
        </div>
    </div>

    <div class="section">
        <h2>4. Podsumowanie i Zbiorcze Porównanie Metod</h2>
        <table>
            <thead>
                <tr>
                    <th>Metoda / Model Klasyfikacji</th>
                    <th>Dokładność Ogólna (Accuracy)</th>
                    <th>Najlepsza Klasa (Wysoki Recall)</th>
                    <th>Najsłabsza Klasa (Niski Recall)</th>
                    <th>Główne Przyczyny Błędów i Ograniczenia</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Metoda Regułowa CV (L3)</strong></td>
                    <td>48.04%</td>
                    <td>EOSINOPHIL</td>
                    <td>LYMPHOCYTE</td>
                    <td>Sztywne progi geometryczne, nachodzenie na siebie sąsiadujących krwinek czerwonych, błędy segmentacji odcieni fioletu.</td>
                </tr>
                <tr>
                    <td><strong>Sieć Konwolucyjna CNN (L5)</strong></td>
                    <td>Wyznaczona powyżej</td>
                    <td>NEUTROPHIL</td>
                    <td>MONOCYTE / LYMPHOCYTE</td>
                    <td>Niezbalansowanie klasowe (dominacja neutrofili w dataset), tendencja sieci do wybierania klasy większościowej.</td>
                </tr>
            </tbody>
        </table>
    </div>
</div>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[PIPELINE] SUKCES! Wygenerowano pełny raport sprawozdawczy: {report_path}")


if __name__ == "__main__":
    print("=================== INTEGRACJA POTOKU URUCHOMIENIOWEGO (LIVE LOGS) ===================")

    l3_logs = run_script("L3_pomiary.py")
    l5_logs = run_script("L5_cnn_weighted.py")

    generate_html_report(l3_logs, l5_logs)
    print("======================================================================================")




