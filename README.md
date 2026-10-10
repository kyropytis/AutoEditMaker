# AutoEditMaker (Pro Studio Edition)

Ein professionelles Beat-Sync Video-Editing Tool mit moderner CustomTkinter-Oberfläche, hardwarebeschleunigter FFmpeg-Engine, präziser Librosa-Audioanalyse und TikTok/Reels Impact-Effekten.

## 🚀 Highlights & Features

- **🌀 Spinning Dark Triad / Sigma Overlay**:
  - Eigenes Bild (PNG mit Transparenz oder JPG) einbinden.
  - Dreht sich kontinuierlich um sich selbst im Phonk/Dark-Triad-Style.
  - **Start-Trigger**: Automatisch ab dem Beat-Drop, ab bestimmter Sekunde oder von Anfang an.
  - **Ecken-Modus**: Wechselt zum Beat clean animiert im Kreis zwischen den 4 Ecken (Oben-Links → Oben-Rechts → Unten-Rechts → Unten-Links) oder feste Ecke.
  - **Auto-Kreiszuschnitt** mit leuchtendem Rand.
- **⚡ 60 FPS Support & butterweiches Playback**:
  - Wählbar zwischen 60 FPS und 30 FPS für flüssige Gaming- & Action-Edits.
- **✨ Hochwertige Impact-Effekte (Keine billigen Glitches)**:
  - **💥 Bass-Shake**: Gedämpfter Kamera-Rückstoß auf Kick-Hits.
  - **✨ Bloom-Flash**: Weicher, exponentieller Studio-Lichtblitz.
  - **🔍 Punch-Zoom**: Dynamischer optischer Beat-Punch mit Ease-Out.
  - **🎥 Slow Push-In**: Sanfte filmische Vorwärtsfahrt.
  - **⚡ RGB-Shockwave**: Millisekunden-kurze chromatische Aberration nur im Moment des Aufpralls.
  - **🚀 Speed-Ramp**: Schneller Temposchub ohne Takt-Verlust.
- **🎯 100% Perfekter Beat-Sync & Takt-Magnet**:
  - Manuelle Klicks rasten über den **🧲 Takt-Magneten** millimetergenau an den echten Drum-Peaks ein.
  - Frame-Quantisierung verhindert jeglichen kumulierten Drift.
- **🔊 Audio-Mixing & Sanfter Fade-Out**:
  - Original Clip-Sound stummschalten oder zu 15%/35% dezent beibehalten (für Schüsse/Killsounds).
  - Automatischer 0.8s Fade-Out am Ende gegen abruptes Abreißen.
- **🛑 Abbrechen-Button & Ordner direkt öffnen**:
  - Laufende Rendervorgänge sauber per Knopfdruck abbrechen.
  - Nach Fertigstellung direkt mit einem Klick den Ausgabeordner öffnen oder das Video abspielen.

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
