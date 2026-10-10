# AutoEditMaker

Ein automatisches Beat-Sync Video-Editing Tool mit grafischer Benutzeroberfläche (CustomTkinter), MoviePy und Librosa.

## Features

- **Beat-Erkennung**: Automatische Onset- und Beat-Erkennung über Librosa
- **Synchronisierter Schnitt**: Automatisches Schneiden von Videoclips im Takt der Musik
- **Visuelle Effekte**: Zoom-In/Out, Flashs, Glitch-Effekte, Shake und Geschwindigkeitsrampen
- **Moderne GUI**: Intuitive Dark-Mode Oberfläche mit CustomTkinter

## Installation & Start

### Schnellstart (Windows)
Einfach `start.bat` doppelklicken. Das Skript erstellt automatisch eine virtuelle Umgebung (`venv`), installiert alle nötigen Abhängigkeiten und startet die App.

### Manuelle Installation

1. Virtuelle Umgebung anlegen & aktivieren:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

3. Anwendung starten:
   ```bash
   python app.py
   ```

## Abhängigkeiten

- Python 3.10+
- `customtkinter`
- `moviepy==1.0.3`
- `librosa`
- `numpy`
- `pygame`
