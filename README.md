# AutoEditMaker (Pro Studio Edition)

Ein professionelles Beat-Sync Video-Editing Tool mit moderner CustomTkinter-Oberflaeche, hardwarebeschleunigter FFmpeg-Engine, praeziser Librosa-Audioanalyse und High-Impact Videoeffekten.

## Highlights & Features

- **Profile & Presets (Speichern & Laden)**:
  - Vollstaendige Presets direkt speichern, laden, importieren und exportieren.
  - Vorkonfigurierte Profile fuer TikTok/Reels, Velocity, Cinematic und Minimal.
- **Spinning Bild-Overlay**:
  - Eigenes Bild (PNG mit Transparenz oder JPG) einbinden.
  - Kontinuierliche Rotation ueber Schnittgrenzen hinweg mit Geschwindigkeits- und Deckkraft-Regler.
  - Start-Trigger: Automatisch ab dem Beat-Drop, ab einer festgelegten Sekunde oder von Anfang an.
  - Positionierung: Animierter Eckenwechsel zum Takt, feste Ecken oder zentriert in der Bildmitte.
  - Optionaler Kreiszuschnitt mit weichem Rand.
- **60 FPS Support & fluessiges Playback**:
  - Waehlbar zwischen 60 FPS und 30 FPS fuer fluessige Action- und Gameplay-Edits.
- **Professionelle Impact-Effekte (Keine billigen Glitches)**:
  - **Bass-Shake**: Gedaempfter Kamera-Rueckstoss mit dynamischem Rotations-Tilt auf Kick-Hits.
  - **Exposure Bloom-Flash**: Echter Highlight-Lichtburst mit Belichtungskurve (kein Milcheffekt).
  - **Punch-Zoom**: Dynamischer optischer Beat-Punch mit kubischem Ease-Out.
  - **Radial RGB-Aberration**: Optische chromatische Linsenzerlegung an den Bildraendern.
  - **Cinematic Vignette**: Dunkler Randgradient fuer Fokussierung auf die Bildmitte.
  - **Cinematic S-Curve LUT**: Erhoehter Kontrast und Farbsaettigung fuer satten Look.
  - **Slow Push-In**: Sanfte filmische Kamerafahrt fuer ruhigere Passagen.
  - **Speed-Ramp**: Taktgenaue Geschwindigkeitsrampen.
  - **Effekt-Intensitaet**: Stufenlos von 20% bis 200% anpassbar.
- **Praeziser Beat-Sync & Takt-Magnet**:
  - Multi-Band Transient Tracking (Kicks und Snares getrennt) mit Attack-Backtracking.
  - Manuelle Klicks rasten ueber den Takt-Magneten praezise an den echten Drum-Peaks ein.
  - Frame-Quantisierung verhindert jeglichen kumulierten Drift ueber die gesamte Songlaenge.
- **Audio-Mixing & Fade-Out**:
  - Original Clip-Sound stummschalten oder zu 15%/35% dezent beibehalten (fuer In-Game Sounds/SFX).
  - Automatischer 0.8s Fade-Out am Ende gegen abruptes Abreissen.
- **Prozesskontrolle & Schnellzugriff**:
  - Laufende Rendervorgaenge sauber per Knopfdruck abbrechen.
  - Nach Fertigstellung direkt den Ausgabeordner oeffnen oder das Video abspielen.

## Installation & Start

### Schnellstart (Windows)
Doppelklick auf `start.bat`. Das Skript erstellt bei Bedarf eine virtuelle Umgebung (`venv`), installiert alle noetigen Abhaengigkeiten und startet die App.

### Manuelle Installation

1. Virtuelle Umgebung anlegen & aktivieren:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

2. Abhaengigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

3. Anwendung starten:
   ```bash
   python app.py
   ```
