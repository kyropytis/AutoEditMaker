import customtkinter as ctk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, ColorClip, vfx, ImageClip
import os
import threading
import random
import traceback
import time
import tempfile
import subprocess
import shutil
import pygame
import imageio_ffmpeg
import PIL.Image
import PIL.ImageDraw
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# Compatibility Patches & OpenCV Optimization
# =====================================================================
if not hasattr(PIL.Image, 'ANTIALIAS'):
    try:
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import moviepy.video.io.ffmpeg_reader as ffr
_orig_read_frame = ffr.FFMPEG_VideoReader.read_frame
def _robust_read_frame(self):
    try:
        return _orig_read_frame(self)
    except IOError as e:
        if "failed to read the first frame" in str(e):
            try:
                self.initialize(0.1)
                self.pos = int(self.fps * 0.1 + 0.00001) + 1
                return _orig_read_frame(self)
            except Exception:
                pass
            w, h = self.size
            depth = getattr(self, 'depth', 3)
            blank = np.zeros((h, w, depth), dtype='uint8')
            self.lastread = blank
            return blank
        raise
ffr.FFMPEG_VideoReader.read_frame = _robust_read_frame

# Patch for MoviePy 1.0.3 crop bug: prevent negative coordinate slicing
import moviepy.video.fx.crop as mpy_crop
def _safe_moviepy_crop(clip, x1=None, y1=None, x2=None, y2=None, width=None, height=None, x_center=None, y_center=None):
    cw, ch = clip.size
    if width is not None and x1 is not None: x2 = x1 + width
    elif width is not None and x2 is not None: x1 = x2 - width
    if height is not None and y1 is not None: y2 = y1 + height
    elif height is not None and y2 is not None: y1 = y2 - height

    if x_center is not None and width is not None:
        x1 = x_center - width / 2.0
        x2 = x_center + width / 2.0
    if y_center is not None and height is not None:
        y1 = y_center - height / 2.0
        y2 = y_center + height / 2.0

    if x1 is None: x1 = 0
    if y1 is None: y1 = 0
    if x2 is None: x2 = cw
    if y2 is None: y2 = ch

    tw_req = int(round(x2 - x1)) if width is None else int(round(width))
    th_req = int(round(y2 - y1)) if height is None else int(round(height))
    tw_req = max(1, min(cw, tw_req))
    th_req = max(1, min(ch, th_req))

    ix1 = max(0, min(cw - tw_req, int(round(x1))))
    iy1 = max(0, min(ch - th_req, int(round(y1))))
    ix2 = ix1 + tw_req
    iy2 = iy1 + th_req

    newclip = clip.fl_image(lambda pic: pic[iy1:iy2, ix1:ix2], apply_to=["mask"])
    newclip.size = (tw_req, th_req)
    newclip.w = tw_req
    newclip.h = th_req
    return newclip
mpy_crop.crop = _safe_moviepy_crop

# Ultra-fast SIMD resizer via OpenCV
import moviepy.video.fx.resize as mpy_resize
def _cv2_resizer(pic, newsize):
    lx, ly = max(1, int(round(newsize[0]))), max(1, int(round(newsize[1])))
    return cv2.resize(pic.astype(np.uint8), (lx, ly), interpolation=cv2.INTER_LINEAR)
mpy_resize.resizer = _cv2_resizer

# Pre-generate Cinematic S-Curve Contrast LUT
_x_lut = np.linspace(0, 1, 256)
_y_lut = 1.0 / (1.0 + np.exp(-9.0 * (_x_lut - 0.5)))
_y_lut = (_y_lut - _y_lut.min()) / (_y_lut.max() - _y_lut.min())
CINEMATIC_LUT = np.clip(_y_lut * 255.0, 0, 255).astype(np.uint8)

# =====================================================================
# Theme & Configuration
# =====================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class EditMakerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Auto Beat-Sync Edit Maker (Pro Studio Edition)")
        self.geometry("820x970")
        self.minsize(780, 820)

        self.video_paths = []
        self.audio_path = None
        self.audio_duration = 0.0
        self.is_preview_playing = False
        self.manual_times = []
        self.is_cancelled = False
        self.last_rendered_path = None
        self.overlay_img_path = None
        self.source_image_cache = {}
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        pygame.mixer.init()

        self._build_ui()

    # =====================================================================
    # UI Building
    # =====================================================================
    def _build_ui(self):
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=15, pady=12)

        # 1. Header
        lbl_title = ctk.CTkLabel(
            self.main_container,
            text="🎬 Auto Beat-Sync Edit Maker",
            font=("Arial", 26, "bold")
        )
        lbl_title.pack(pady=(4, 2))
        
        lbl_sub = ctk.CTkLabel(
            self.main_container,
            text="Pro Studio Engine • Multi-Band Transient Sync • 60 FPS • OpenCV Hardware-Beschleunigung",
            text_color="#38bdf8",
            font=("Arial", 12)
        )
        lbl_sub.pack(pady=(0, 12))

        # 2. Media Selection Frame
        media_frame = ctk.CTkFrame(self.main_container)
        media_frame.pack(fill="x", pady=5)

        header_media = ctk.CTkFrame(media_frame, fg_color="transparent")
        header_media.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(header_media, text="1. Medien (Videos & Bilder)", font=("Arial", 14, "bold")).pack(side="left")
        self.lbl_videos_count = ctk.CTkLabel(header_media, text="0 Dateien ausgewählt", text_color="gray")
        self.lbl_videos_count.pack(side="right")

        btn_row_media = ctk.CTkFrame(media_frame, fg_color="transparent")
        btn_row_media.pack(fill="x", padx=12, pady=4)

        self.btn_select_videos = ctk.CTkButton(
            btn_row_media,
            text="📁 Clips / Bilder wählen",
            command=self.select_videos,
            width=180
        )
        self.btn_select_videos.pack(side="left", padx=(0, 10))

        self.btn_clear_videos = ctk.CTkButton(
            btn_row_media,
            text="Zurücksetzen",
            command=self.clear_videos,
            fg_color="#444444",
            hover_color="#555555",
            width=100
        )
        self.btn_clear_videos.pack(side="left", padx=5)

        self.shuffle_var = ctk.BooleanVar(value=True)
        self.chk_shuffle = ctk.CTkCheckBox(
            btn_row_media,
            text="Zufällige Reihenfolge (Shuffle)",
            variable=self.shuffle_var
        )
        self.chk_shuffle.pack(side="right", padx=5)

        self.media_listbox = ctk.CTkTextbox(media_frame, height=65, font=("Consolas", 11), activate_scrollbars=True)
        self.media_listbox.pack(fill="x", padx=12, pady=(4, 10))
        self.media_listbox.insert("1.0", "Noch keine Dateien gewählt. Unterstützt MP4, MOV, MKV, AVI, PNG, JPG, WEBP...")
        self.media_listbox.configure(state="disabled")

        # 3. Audio Selection & Trim Frame
        audio_frame = ctk.CTkFrame(self.main_container)
        audio_frame.pack(fill="x", pady=5)

        header_audio = ctk.CTkFrame(audio_frame, fg_color="transparent")
        header_audio.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(header_audio, text="2. Musik (Beat & Song)", font=("Arial", 14, "bold")).pack(side="left")
        self.lbl_audio_status = ctk.CTkLabel(header_audio, text="Kein Song gewählt", text_color="gray")
        self.lbl_audio_status.pack(side="right")

        btn_row_audio = ctk.CTkFrame(audio_frame, fg_color="transparent")
        btn_row_audio.pack(fill="x", padx=12, pady=4)

        self.btn_select_audio = ctk.CTkButton(
            btn_row_audio,
            text="🎵 Song auswählen",
            command=self.select_audio,
            width=180
        )
        self.btn_select_audio.pack(side="left", padx=(0, 10))

        self.btn_audio_preview = ctk.CTkButton(
            btn_row_audio,
            text="▶ Vorhören",
            command=self.toggle_audio_preview,
            fg_color="#2b719e",
            hover_color="#1f5375",
            width=110,
            state="disabled"
        )
        self.btn_audio_preview.pack(side="left", padx=5)

        slider_frame = ctk.CTkFrame(audio_frame, fg_color="transparent")
        slider_frame.pack(fill="x", padx=12, pady=(4, 10))

        start_row = ctk.CTkFrame(slider_frame, fg_color="transparent")
        start_row.pack(fill="x", pady=2)
        self.lbl_start = ctk.CTkLabel(start_row, text="Start: 0.0s", width=90, anchor="w", font=("Arial", 12))
        self.lbl_start.pack(side="left")
        self.slider_start = ctk.CTkSlider(start_row, from_=0, to=100, command=self.update_start_lbl)
        self.slider_start.set(0)
        self.slider_start.pack(side="right", fill="x", expand=True, padx=5)

        end_row = ctk.CTkFrame(slider_frame, fg_color="transparent")
        end_row.pack(fill="x", pady=2)
        self.lbl_end = ctk.CTkLabel(end_row, text="Ende: 0.0s", width=90, anchor="w", font=("Arial", 12))
        self.lbl_end.pack(side="left")
        self.slider_end = ctk.CTkSlider(end_row, from_=0, to=100, command=self.update_end_lbl)
        self.slider_end.set(0)
        self.slider_end.pack(side="right", fill="x", expand=True, padx=5)

        # 4. Beat Detection & Pacing Frame (Upgraded Multi-Band Engine)
        sync_frame = ctk.CTkFrame(self.main_container)
        sync_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(sync_frame, text="3. Beat-Sync & Multi-Band Timing", font=("Arial", 14, "bold")).pack(anchor="w", padx=12, pady=(10, 3))

        mode_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        mode_row.pack(fill="x", padx=12, pady=3)

        self.mode_var = ctk.StringVar(value="auto")
        ctk.CTkRadioButton(mode_row, text="Auto Takt (Kick & Snare Transient-Sync)", variable=self.mode_var, value="auto").pack(side="left", padx=8)
        ctk.CTkRadioButton(mode_row, text="Dynamisch (Alle Onsets)", variable=self.mode_var, value="onsets").pack(side="left", padx=8)
        ctk.CTkRadioButton(mode_row, text="Manuell (Mit Takt-Magnet)", variable=self.mode_var, value="manual").pack(side="left", padx=8)

        pace_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        pace_row.pack(fill="x", padx=12, pady=(4, 10))

        ctk.CTkLabel(pace_row, text="Schnitt-Tempo:").pack(side="left", padx=(0, 8))
        self.pace_var = ctk.StringVar(value="1x (Jeder Takt)")
        self.combo_pace = ctk.CTkComboBox(
            pace_row,
            values=["1x (Jeder Takt)", "2x Schnell (Halber Takt)", "0.5x Entspannt (Doppel-Takt)"],
            variable=self.pace_var,
            width=210
        )
        self.combo_pace.pack(side="left", padx=5)

        # 5. Format, Skalierung & Qualität
        format_frame = ctk.CTkFrame(self.main_container)
        format_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(format_frame, text="4. Format, Auflösung, Qualität & FPS", font=("Arial", 14, "bold")).pack(anchor="w", padx=12, pady=(10, 3))
        
        row_fmt = ctk.CTkFrame(format_frame, fg_color="transparent")
        row_fmt.pack(fill="x", padx=12, pady=3)

        ctk.CTkLabel(row_fmt, text="Format:").pack(side="left", padx=(0, 8))
        self.orient_var = ctk.StringVar(value="original")
        ctk.CTkRadioButton(row_fmt, text="Originalformat (Kein Zoom)", variable=self.orient_var, value="original").pack(side="left", padx=6)
        ctk.CTkRadioButton(row_fmt, text="📱 Hochkant (9:16)", variable=self.orient_var, value="portrait").pack(side="left", padx=6)
        ctk.CTkRadioButton(row_fmt, text="🖥 Quer (16:9)", variable=self.orient_var, value="landscape").pack(side="left", padx=6)

        row_scale_fps = ctk.CTkFrame(format_frame, fg_color="transparent")
        row_scale_fps.pack(fill="x", padx=12, pady=3)

        ctk.CTkLabel(row_scale_fps, text="Einpassung:").pack(side="left", padx=(0, 6))
        self.scale_mode_var = ctk.StringVar(value="fit")
        ctk.CTkRadioButton(row_scale_fps, text="Ganzes Bild (Fit)", variable=self.scale_mode_var, value="fit").pack(side="left", padx=5)
        ctk.CTkRadioButton(row_scale_fps, text="Vollbild (Crop)", variable=self.scale_mode_var, value="fill").pack(side="left", padx=5)

        ctk.CTkLabel(row_scale_fps, text="Framerate:").pack(side="left", padx=(15, 6))
        self.fps_var = ctk.IntVar(value=60)
        ctk.CTkRadioButton(row_scale_fps, text="60 FPS (Flüssig)", variable=self.fps_var, value=60).pack(side="left", padx=5)
        ctk.CTkRadioButton(row_scale_fps, text="30 FPS", variable=self.fps_var, value=30).pack(side="left", padx=5)

        row_quality = ctk.CTkFrame(format_frame, fg_color="transparent")
        row_quality.pack(fill="x", padx=12, pady=(3, 10))

        ctk.CTkLabel(row_quality, text="Export-Qualität:").pack(side="left", padx=(0, 6))
        self.quality_var = ctk.StringVar(value="💎 High Quality (CRF 16 Studio)")
        ctk.CTkComboBox(
            row_quality,
            values=[
                "💎 High Quality (CRF 16 Studio)",
                "👑 Maximal (CRF 14 Glasklar)",
                "⚡ Ultra-Fast (CRF 20 Schnell)"
            ],
            variable=self.quality_var,
            width=230
        ).pack(side="left", padx=5)

        ctk.CTkLabel(row_quality, text="Auflösung:").pack(side="left", padx=(15, 6))
        self.res_var = ctk.StringVar(value="1080p (Full HD)")
        ctk.CTkComboBox(
            row_quality,
            values=[
                "1080p (Full HD)",
                "1440p (2K Ultra Sharp)",
                "Originalgröße der Dateien"
            ],
            variable=self.res_var,
            width=180
        ).pack(side="left", padx=5)

        # 6. Audio-Mixing & Fade-Out
        audio_opt_frame = ctk.CTkFrame(self.main_container)
        audio_opt_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(audio_opt_frame, text="5. Audio-Mixing & Ausblenden", font=("Arial", 14, "bold")).pack(anchor="w", padx=12, pady=(10, 3))
        
        row_mix = ctk.CTkFrame(audio_opt_frame, fg_color="transparent")
        row_mix.pack(fill="x", padx=12, pady=(3, 10))

        ctk.CTkLabel(row_mix, text="Original Clip-Sound:").pack(side="left", padx=(0, 8))
        self.clip_audio_var = ctk.StringVar(value="Stumm (100% Musik)")
        ctk.CTkComboBox(
            row_mix,
            values=["Stumm (100% Musik)", "Dezent 15% (Killsounds/SFX)", "Mittel 35% (Dialoge/SFX)"],
            variable=self.clip_audio_var,
            width=210
        ).pack(side="left", padx=5)

        self.fade_out_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            row_mix,
            text="Sanfter Audio- & Video-Fade-Out am Ende (0.8s)",
            variable=self.fade_out_var
        ).pack(side="left", padx=(15, 5))

        # 7. Modern High-Impact Effects & Cinematic Grading Frame
        effects_frame = ctk.CTkFrame(self.main_container)
        effects_frame.pack(fill="x", pady=5)

        eff_header = ctk.CTkFrame(effects_frame, fg_color="transparent")
        eff_header.pack(fill="x", padx=12, pady=(10, 3))
        ctk.CTkLabel(eff_header, text="6. Professionelle Impact-Effekte & Cinematic Look", font=("Arial", 14, "bold")).pack(side="left")

        preset_row = ctk.CTkFrame(effects_frame, fg_color="transparent")
        preset_row.pack(fill="x", padx=12, pady=3)

        ctk.CTkLabel(preset_row, text="Effekt-Preset:").pack(side="left", padx=(0, 8))
        self.eff_preset_var = ctk.StringVar(value="Cinematic Flow (Punch, Shake & Warm Bloom)")
        self.combo_preset = ctk.CTkComboBox(
            preset_row,
            values=[
                "Cinematic Flow (Punch, Shake & Warm Bloom)",
                "Hype & Action (Alle Effekte aktiv)",
                "Clean / Minimal (Nur Cuts & dezent)",
                "Benutzerdefiniert"
            ],
            variable=self.eff_preset_var,
            command=self.on_preset_changed,
            width=290
        )
        self.combo_preset.pack(side="left", padx=5)

        self.fx_cinematic_lut = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            preset_row,
            text="🎬 Cinematic S-Curve Contrast & Color Pop",
            variable=self.fx_cinematic_lut
        ).pack(side="left", padx=(15, 5))

        eff_grid = ctk.CTkFrame(effects_frame, fg_color="transparent")
        eff_grid.pack(fill="x", padx=12, pady=(6, 10))

        self.fx_shake = ctk.BooleanVar(value=True)
        self.fx_bloom_flash = ctk.BooleanVar(value=True)
        self.fx_punch_zoom = ctk.BooleanVar(value=True)
        self.fx_slow_push = ctk.BooleanVar(value=True)
        self.fx_rgb_shock = ctk.BooleanVar(value=True)
        self.fx_speedramp = ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(eff_grid, text="💥 Bass-Shake (OpenCV Recoil)", variable=self.fx_shake, command=self._set_custom_preset).grid(row=0, column=0, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="✨ Studio Bloom-Flash (Warm)", variable=self.fx_bloom_flash, command=self._set_custom_preset).grid(row=0, column=1, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="🔍 Punch-Zoom (Optischer Ease)", variable=self.fx_punch_zoom, command=self._set_custom_preset).grid(row=0, column=2, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="🎥 Slow Push-In (Sanfte Fahrt)", variable=self.fx_slow_push, command=self._set_custom_preset).grid(row=1, column=0, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="⚡ RGB-Shockwave (Aberration)", variable=self.fx_rgb_shock, command=self._set_custom_preset).grid(row=1, column=1, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="🚀 Speed-Ramp (Sync-Safe)", variable=self.fx_speedramp, command=self._set_custom_preset).grid(row=1, column=2, sticky="w", padx=8, pady=3)

        # 8. Dark Triad Spinning Overlay Frame
        overlay_frame = ctk.CTkFrame(self.main_container)
        overlay_frame.pack(fill="x", pady=5)

        overlay_header = ctk.CTkFrame(overlay_frame, fg_color="transparent")
        overlay_header.pack(fill="x", padx=12, pady=(10, 4))

        self.enable_overlay_var = ctk.BooleanVar(value=False)
        self.chk_overlay = ctk.CTkCheckBox(
            overlay_header,
            text="🌀 Spinning Overlay aktivieren (Dark Triad / Sigma Style)",
            variable=self.enable_overlay_var,
            command=self._toggle_overlay_ui,
            font=("Arial", 14, "bold")
        )
        self.chk_overlay.pack(side="left")

        self.overlay_options_frame = ctk.CTkFrame(overlay_frame, fg_color="transparent")

        # Row 1: Image Selection
        row_ov1 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov1.pack(fill="x", padx=12, pady=4)

        self.btn_select_overlay = ctk.CTkButton(
            row_ov1,
            text="🖼 Overlay-Bild wählen (PNG/JPG)",
            command=self.select_overlay_image,
            width=220
        )
        self.btn_select_overlay.pack(side="left", padx=(0, 10))

        self.lbl_overlay_name = ctk.CTkLabel(row_ov1, text="Kein Bild ausgewählt", text_color="gray")
        self.lbl_overlay_name.pack(side="left")

        # Row 2: Timing / Start-Point
        row_ov2 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov2.pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(row_ov2, text="Start-Zeitpunkt:").pack(side="left", padx=(0, 8))
        self.overlay_start_mode = ctk.StringVar(value="drop")
        ctk.CTkRadioButton(row_ov2, text="🔥 Ab Beat-Drop (Automatisch)", variable=self.overlay_start_mode, value="drop").pack(side="left", padx=6)
        ctk.CTkRadioButton(row_ov2, text="⏱ Ab bestimmter Zeit", variable=self.overlay_start_mode, value="custom").pack(side="left", padx=6)
        ctk.CTkRadioButton(row_ov2, text="♾ Von Anfang an", variable=self.overlay_start_mode, value="start").pack(side="left", padx=6)

        self.row_ov_slider = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        self.row_ov_slider.pack(fill="x", padx=12, pady=2)
        self.lbl_overlay_sec = ctk.CTkLabel(self.row_ov_slider, text="Start bei: 10.0s", width=110, anchor="w")
        self.lbl_overlay_sec.pack(side="left")
        self.slider_overlay_sec = ctk.CTkSlider(self.row_ov_slider, from_=0, to=60, command=lambda v: self.lbl_overlay_sec.configure(text=f"Start bei: {v:.1f}s"))
        self.slider_overlay_sec.set(10.0)
        self.slider_overlay_sec.pack(side="right", fill="x", expand=True, padx=5)

        # Row 3: Placement & Size
        row_ov3 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov3.pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(row_ov3, text="Platzierung:").pack(side="left", padx=(0, 6))
        self.overlay_corner_var = ctk.StringVar(value="Wechselt zum Takt (Animiert)")
        self.combo_corner = ctk.CTkComboBox(
            row_ov3,
            values=[
                "Wechselt zum Takt (Animiert)",
                "🎯 In der Mitte (Zentriert)",
                "Oben Links",
                "Oben Rechts",
                "Unten Links",
                "Unten Rechts"
            ],
            variable=self.overlay_corner_var,
            width=220
        )
        self.combo_corner.pack(side="left", padx=5)

        ctk.CTkLabel(row_ov3, text="Größe:").pack(side="left", padx=(12, 6))
        self.overlay_size_var = ctk.StringVar(value="Mittel (30% - Standard)")
        self.combo_ov_size = ctk.CTkComboBox(
            row_ov3,
            values=[
                "Kompakt (20% - Dezent)",
                "Mittel (30% - Standard)",
                "Groß (45% - Fett & Dominant)"
            ],
            variable=self.overlay_size_var,
            width=210
        )
        self.combo_ov_size.pack(side="left", padx=5)

        self.overlay_circle_cut_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            row_ov3,
            text="Kreis-Zuschnitt mit Rand",
            variable=self.overlay_circle_cut_var
        ).pack(side="left", padx=(12, 5))

        # Row 4: Speed & Opacity Slider
        row_ov4 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov4.pack(fill="x", padx=12, pady=(4, 12))

        speed_col = ctk.CTkFrame(row_ov4, fg_color="transparent")
        speed_col.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.lbl_ov_speed = ctk.CTkLabel(speed_col, text="Dreh-Speed: 1.2x (Umdr./s)", anchor="w")
        self.lbl_ov_speed.pack(fill="x")
        self.slider_ov_speed = ctk.CTkSlider(
            speed_col, from_=0.2, to=4.0,
            command=lambda v: self.lbl_ov_speed.configure(text=f"Dreh-Speed: {v:.1f}x (Umdr./s)")
        )
        self.slider_ov_speed.set(1.2)
        self.slider_ov_speed.pack(fill="x", pady=2)

        opacity_col = ctk.CTkFrame(row_ov4, fg_color="transparent")
        opacity_col.pack(side="right", fill="x", expand=True, padx=(10, 0))
        self.lbl_ov_opacity = ctk.CTkLabel(opacity_col, text="Deckkraft: 100%", anchor="w")
        self.lbl_ov_opacity.pack(fill="x")
        self.slider_ov_opacity = ctk.CTkSlider(
            opacity_col, from_=10, to=100,
            command=lambda v: self.lbl_ov_opacity.configure(text=f"Deckkraft: {int(v)}%")
        )
        self.slider_ov_opacity.set(100.0)
        self.slider_ov_opacity.pack(fill="x", pady=2)

        # 9. Render & Controls Frame
        bottom_frame = ctk.CTkFrame(self.main_container)
        bottom_frame.pack(fill="x", pady=8)

        btn_row_action = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        btn_row_action.pack(fill="x", padx=20, pady=(12, 6))

        self.btn_generate = ctk.CTkButton(
            btn_row_action,
            text="🚀 Video Rendern (Pro Studio Engine)",
            command=self.start_generation,
            fg_color="#1f8a4c",
            hover_color="#176b3b",
            height=44,
            font=("Arial", 16, "bold")
        )
        self.btn_generate.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_cancel = ctk.CTkButton(
            btn_row_action,
            text="🛑 Abbrechen",
            command=self.cancel_generation,
            fg_color="#8b1e1e",
            hover_color="#631414",
            height=44,
            width=120,
            font=("Arial", 14, "bold"),
            state="disabled"
        )
        self.btn_cancel.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(bottom_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=(4, 6))
        self.progress_bar.set(0.0)

        status_row = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        status_row.pack(fill="x", padx=20, pady=(0, 6))

        self.lbl_status = ctk.CTkLabel(status_row, text="Bereit für Schnitt", font=("Arial", 12), text_color="yellow")
        self.lbl_status.pack(side="left")

        self.lbl_pct = ctk.CTkLabel(status_row, text="0%", font=("Arial", 12, "bold"))
        self.lbl_pct.pack(side="right")

        self.post_action_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        self.btn_open_folder = ctk.CTkButton(
            self.post_action_frame,
            text="📂 Ordner im Explorer öffnen",
            command=self.open_output_folder,
            fg_color="#2b719e",
            hover_color="#1f5375",
            width=200
        )
        self.btn_open_folder.pack(side="left", padx=5)

        self.btn_play_video = ctk.CTkButton(
            self.post_action_frame,
            text="▶ Video abspielen",
            command=self.play_output_video,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            width=150
        )
        self.btn_play_video.pack(side="left", padx=5)

    def _toggle_overlay_ui(self):
        if self.enable_overlay_var.get():
            self.overlay_options_frame.pack(fill="x", padx=5, pady=(0, 8))
        else:
            self.overlay_options_frame.pack_forget()

    def select_overlay_image(self):
        path = filedialog.askopenfilename(
            title="Wähle Bild für Spinning Overlay",
            filetypes=[("Bilder", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Alle Dateien", "*.*")]
        )
        if path:
            self.overlay_img_path = path
            self.lbl_overlay_name.configure(text=os.path.basename(path), text_color="#38bdf8")

    # =====================================================================
    # Preset & Effect Helpers
    # =====================================================================
    def on_preset_changed(self, choice):
        if "Cinematic Flow" in choice:
            self.fx_shake.set(True)
            self.fx_bloom_flash.set(True)
            self.fx_punch_zoom.set(True)
            self.fx_slow_push.set(True)
            self.fx_rgb_shock.set(False)
            self.fx_speedramp.set(True)
            self.fx_cinematic_lut.set(True)
        elif "Hype & Action" in choice:
            self.fx_shake.set(True)
            self.fx_bloom_flash.set(True)
            self.fx_punch_zoom.set(True)
            self.fx_slow_push.set(True)
            self.fx_rgb_shock.set(True)
            self.fx_speedramp.set(True)
            self.fx_cinematic_lut.set(True)
        elif "Clean / Minimal" in choice:
            self.fx_shake.set(False)
            self.fx_bloom_flash.set(True)
            self.fx_punch_zoom.set(False)
            self.fx_slow_push.set(False)
            self.fx_rgb_shock.set(False)
            self.fx_speedramp.set(False)
            self.fx_cinematic_lut.set(False)

    def _set_custom_preset(self):
        self.eff_preset_var.set("Benutzerdefiniert")

    # =====================================================================
    # Thread-Safe UI Update Helpers
    # =====================================================================
    def set_status(self, text, progress=None):
        def _update():
            self.lbl_status.configure(text=text)
            if progress is not None:
                p_clamped = max(0.0, min(1.0, float(progress)))
                self.progress_bar.set(p_clamped)
                self.lbl_pct.configure(text=f"{int(p_clamped * 100)}%")
        self.after(0, _update)

    def show_info(self, title, msg):
        self.after(0, lambda: messagebox.showinfo(title, msg))

    def show_error(self, title, msg):
        self.after(0, lambda: messagebox.showerror(title, msg))

    def show_warning(self, title, msg):
        self.after(0, lambda: messagebox.showwarning(title, msg))

    # =====================================================================
    # Post-Export Actions
    # =====================================================================
    def open_output_folder(self):
        if self.last_rendered_path and os.path.exists(self.last_rendered_path):
            norm_p = os.path.normpath(self.last_rendered_path)
            subprocess.Popen(f'explorer /select,"{norm_p}"')

    def play_output_video(self):
        if self.last_rendered_path and os.path.exists(self.last_rendered_path):
            try:
                os.startfile(self.last_rendered_path)
            except Exception as e:
                self.show_error("Fehler", f"Konnte Video nicht abspielen:\n{e}")

    # =====================================================================
    # Sliders & Audio Preview
    # =====================================================================
    def update_start_lbl(self, val):
        self.lbl_start.configure(text=f"Start: {val:.1f}s")
        if val > self.slider_end.get():
            self.slider_end.set(val)
            self.lbl_end.configure(text=f"Ende: {val:.1f}s")
        self.stop_audio_preview()

    def update_end_lbl(self, val):
        self.lbl_end.configure(text=f"Ende: {val:.1f}s")
        if val < self.slider_start.get():
            self.slider_start.set(val)
            self.lbl_start.configure(text=f"Start: {val:.1f}s")
        self.stop_audio_preview()

    def toggle_audio_preview(self):
        if self.is_preview_playing:
            self.stop_audio_preview()
        else:
            self.start_audio_preview()

    def start_audio_preview(self):
        if not self.audio_path or not os.path.exists(self.audio_path):
            return
        start_t = self.slider_start.get()
        end_t = self.slider_end.get()
        if start_t >= end_t:
            return

        try:
            pygame.mixer.music.load(self.audio_path)
            pygame.mixer.music.play(start=start_t)
            self.is_preview_playing = True
            self.btn_audio_preview.configure(text="⏸ Stoppen", fg_color="#b83232", hover_color="#852424")
            duration = end_t - start_t
            threading.Thread(target=self._preview_watchdog, args=(duration,), daemon=True).start()
        except Exception as e:
            self.stop_audio_preview()
            self.show_error("Preview Fehler", f"Konnte Audio-Vorschau nicht starten:\n{e}")

    def _preview_watchdog(self, duration):
        t0 = time.perf_counter()
        while self.is_preview_playing:
            time.sleep(0.05)
            if time.perf_counter() - t0 >= duration:
                self.after(0, self.stop_audio_preview)
                break

    def stop_audio_preview(self):
        if self.is_preview_playing:
            self.is_preview_playing = False
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        self.btn_audio_preview.configure(text="▶ Vorhören", fg_color="#2b719e", hover_color="#1f5375")

    # =====================================================================
    # Media & Audio Selection
    # =====================================================================
    def select_videos(self):
        paths = filedialog.askopenfilenames(
            title="Wähle Medien (Videos & Fotos)",
            filetypes=[
                ("Medien-Dateien", "*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg *.webp *.bmp"),
                ("Alle Dateien", "*.*")
            ]
        )
        if paths:
            self.video_paths = list(paths)
            self.source_image_cache.clear()
            self.update_media_listbox()

    def clear_videos(self):
        self.video_paths = []
        self.source_image_cache.clear()
        self.update_media_listbox()

    def update_media_listbox(self):
        self.media_listbox.configure(state="normal")
        self.media_listbox.delete("1.0", "end")
        count = len(self.video_paths)
        if count == 0:
            self.lbl_videos_count.configure(text="0 Dateien ausgewählt", text_color="gray")
            self.media_listbox.insert("1.0", "Noch keine Dateien gewählt.")
        else:
            self.lbl_videos_count.configure(text=f"{count} Dateien geladen", text_color="white")
            lines = [f"{i+1:02d}. {os.path.basename(p)}" for i, p in enumerate(self.video_paths)]
            self.media_listbox.insert("1.0", "\n".join(lines))
        self.media_listbox.configure(state="disabled")

    def select_audio(self):
        self.stop_audio_preview()
        path = filedialog.askopenfilename(
            title="Song / Beat auswählen",
            filetypes=[("Audio-Dateien", "*.mp3 *.wav *.ogg *.m4a *.aac *.flac"), ("Alle Dateien", "*.*")]
        )
        if path:
            self.audio_path = path
            filename = os.path.basename(path)
            self.lbl_audio_status.configure(text=filename, text_color="white")
            self.btn_audio_preview.configure(state="normal")

            try:
                clip = AudioFileClip(self.audio_path)
                self.audio_duration = float(clip.duration)
                clip.close()

                self.slider_start.configure(to=self.audio_duration)
                self.slider_end.configure(to=self.audio_duration)
                self.slider_start.set(0)
                self.slider_end.set(self.audio_duration)
                self.update_start_lbl(0)
                self.update_end_lbl(self.audio_duration)
                self.slider_overlay_sec.configure(to=self.audio_duration)
            except Exception as e:
                self.show_error("Fehler", f"Konnte Audio nicht analysieren:\n{e}")

    # =====================================================================
    # Generation Trigger & Cancel
    # =====================================================================
    def cancel_generation(self):
        self.is_cancelled = True
        self.btn_cancel.configure(state="disabled")
        self.set_status("Abbrechen angefordert...", None)

    def start_generation(self):
        self.stop_audio_preview()
        self.is_cancelled = False
        self.post_action_frame.pack_forget()

        if not self.video_paths:
            messagebox.showwarning("Hinweis", "Bitte wähle mindestens ein Video oder Bild aus.")
            return
        if not self.audio_path:
            messagebox.showwarning("Hinweis", "Bitte wähle einen Song aus.")
            return
        if self.enable_overlay_var.get() and (not self.overlay_img_path or not os.path.exists(self.overlay_img_path)):
            messagebox.showwarning("Hinweis", "Du hast das Spinning Overlay aktiviert, aber noch kein Overlay-Bild ausgewählt.")
            return

        start_t = self.slider_start.get()
        end_t = self.slider_end.get()

        if start_t >= end_t:
            messagebox.showwarning("Hinweis", "Die Endzeit muss größer als die Startzeit sein.")
            return

        self.set_status("Prüfe Mediendateien...", 0.02)
        valid_paths = []
        bad_paths = []
        for vp in self.video_paths:
            ext = os.path.splitext(vp)[1].lower()
            try:
                if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                    with PIL.Image.open(vp) as img:
                        img.verify()
                else:
                    test = VideoFileClip(vp)
                    _ = test.get_frame(0)
                    test.close()
                valid_paths.append(vp)
            except Exception:
                bad_paths.append(os.path.basename(vp))

        if bad_paths:
            messagebox.showwarning(
                "Defekte Dateien übersprungen",
                "Folgende Dateien konnten nicht geladen werden und werden übersprungen:\n\n" + "\n".join(bad_paths)
            )

        if not valid_paths:
            messagebox.showerror("Fehler", "Keine lesbaren Mediendateien vorhanden.")
            self.set_status("Bereit für Schnitt", 0.0)
            return

        self.video_paths = valid_paths
        if self.shuffle_var.get():
            random.shuffle(self.video_paths)
        self.update_media_listbox()

        out_path = filedialog.asksaveasfilename(
            title="Video speichern unter",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if not out_path:
            self.set_status("Bereit für Schnitt", 0.0)
            return

        self.btn_generate.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        mode = self.mode_var.get()

        if mode in ("auto", "onsets"):
            self.set_status("Starte Schnitt-Pipeline...", 0.05)
            threading.Thread(
                target=self.generate_edit,
                args=(start_t, end_t, out_path, None, mode),
                daemon=True
            ).start()
        else:
            self.start_manual_recording(start_t, end_t, out_path)

    # =====================================================================
    # Manual Beat Recording (With Takt-Magnet)
    # =====================================================================
    def start_manual_recording(self, start_t, end_t, out_path):
        self.set_status("Bereite Audio & Takt-Magnet vor...", 0.05)
        total_duration = end_t - start_t

        temp_dir = tempfile.mkdtemp(prefix="autoedit_manual_")
        master_audio_wav = os.path.join(temp_dir, "master_manual.wav")

        try:
            cmd = [
                self.ffmpeg_exe, "-y",
                "-ss", f"{start_t:.3f}",
                "-i", self.audio_path,
                "-t", f"{total_duration:.3f}",
                "-vn",
                "-c:a", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                master_audio_wav
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise RuntimeError("FFmpeg Audio-Extraktion fehlgeschlagen.")
        except Exception as e:
            self.show_error("Fehler", f"Audio-Vorbereitung fehlgeschlagen:\n{e}")
            self.btn_generate.configure(state="normal")
            self.btn_cancel.configure(state="disabled")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        detected_onsets = []
        try:
            y_m, sr_m = librosa.load(master_audio_wav, sr=22050)
            y_perc = librosa.effects.percussive(y_m, margin=2.0)
            env_low = librosa.onset.onset_strength(y=y_perc, sr=sr_m, fmax=280)
            env_high = librosa.onset.onset_strength(y=y_perc, sr=sr_m, fmin=800, fmax=5500)
            onset_env = env_low * 1.0 + env_high * 0.65
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_m, backtrack=True, wait=6, delta=0.07)
            detected_onsets = list(librosa.frames_to_time(onset_frames, sr=sr_m))
            del y_m, y_perc, env_low, env_high, onset_env
        except Exception:
            detected_onsets = []

        self.manual_times = []
        rec_win = ctk.CTkToplevel(self)
        rec_win.title("Manuelles Beat-Recording (Mit Takt-Magnet)")
        rec_win.geometry("520x330")
        rec_win.transient(self)
        rec_win.grab_set()

        lbl_info = ctk.CTkLabel(
            rec_win,
            text="Tippe im Takt der Musik: LEERTASTE, Pfeiltasten oder Maus!\nFenster schließen, wenn fertig.",
            font=("Arial", 14)
        )
        lbl_info.pack(pady=12)

        magnet_var = ctk.BooleanVar(value=True)
        chk_magnet = ctk.CTkCheckBox(
            rec_win,
            text="🧲 Takt-Magnet aktiv (Klicks millimetergenau an Drum-Hits einrasten)",
            variable=magnet_var
        )
        chk_magnet.pack(pady=(0, 8))

        lbl_cuts = ctk.CTkLabel(rec_win, text="Aufgenommene Beats: 0", font=("Arial", 18, "bold"), text_color="yellow")
        lbl_cuts.pack(pady=3)

        lbl_time = ctk.CTkLabel(rec_win, text="Zeit: 0.0s / 0.0s", font=("Arial", 12), text_color="gray")
        lbl_time.pack(pady=2)

        try:
            pygame.mixer.music.load(master_audio_wav)
        except Exception as e:
            self.show_error("Fehler", f"Pygame Audio-Ladefehler:\n{e}")
            rec_win.destroy()
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.btn_generate.configure(state="normal")
            self.btn_cancel.configure(state="disabled")
            return

        is_playing = False
        start_perf_time = 0.0

        def record_cut(event=None):
            nonlocal is_playing, start_perf_time
            if not is_playing:
                return
            raw_time = time.perf_counter() - start_perf_time
            compensated_time = max(0.01, raw_time - 0.07)
            if 0.05 < compensated_time < total_duration:
                self.manual_times.append(compensated_time)
                lbl_cuts.configure(text=f"Aufgenommene Beats: {len(self.manual_times)}")

        for key in ("<space>", "<Left>", "<Right>", "<Up>", "<Down>"):
            rec_win.bind(key, record_cut)
        rec_win.bind("<Button-1>", record_cut)
        rec_win.bind("<Button-3>", record_cut)

        def update_playback_clock():
            if is_playing:
                elapsed = time.perf_counter() - start_perf_time
                lbl_time.configure(text=f"Zeit: {min(elapsed, total_duration):.1f}s / {total_duration:.1f}s")
                if elapsed >= total_duration:
                    on_close()
                    return
                rec_win.after(40, update_playback_clock)

        def start_play():
            nonlocal is_playing, start_perf_time
            is_playing = True
            btn_start.configure(state="disabled")
            start_perf_time = time.perf_counter()
            pygame.mixer.music.play()
            rec_win.focus_set()
            update_playback_clock()

        btn_start = ctk.CTkButton(rec_win, text="▶ Aufnahme Starten", command=start_play, font=("Arial", 15, "bold"), height=40)
        btn_start.pack(pady=10)

        def on_close():
            nonlocal is_playing
            is_playing = False
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            rec_win.destroy()

            final_beats = []
            if len(self.manual_times) > 0:
                raw_sorted = sorted(list(set(self.manual_times)))
                if magnet_var.get() and len(detected_onsets) > 0:
                    onsets_arr = np.array(detected_onsets)
                    for tap in raw_sorted:
                        diffs = np.abs(onsets_arr - tap)
                        nearest_idx = int(np.argmin(diffs))
                        if diffs[nearest_idx] <= 0.22:
                            final_beats.append(float(onsets_arr[nearest_idx]))
                        else:
                            final_beats.append(float(tap))
                else:
                    final_beats = raw_sorted

                final_beats = sorted(list(set(final_beats)))
                if final_beats[0] > 0.1:
                    final_beats.insert(0, 0.0)
                else:
                    final_beats[0] = 0.0

                if final_beats[-1] < total_duration:
                    final_beats.append(total_duration)

                self.set_status("Generiere Video aus manuellen Schnitten...", 0.1)
                threading.Thread(
                    target=self.generate_edit,
                    args=(start_t, end_t, out_path, final_beats, "manual", master_audio_wav),
                    daemon=True
                ).start()
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.set_status("Aufnahme abgebrochen (keine Schnitte erfasst).", 0.0)
                self.btn_generate.configure(state="normal")
                self.btn_cancel.configure(state="disabled")

        rec_win.protocol("WM_DELETE_WINDOW", on_close)

    # =====================================================================
    # Main Generation Pipeline (Pro Studio Engine)
    # =====================================================================
    def generate_edit(self, start_t, end_t, out_path, custom_beat_times=None, mode="auto", pre_master_audio=None):
        temp_dir = tempfile.mkdtemp(prefix="autoedit_render_")
        temp_files = []
        try:
            total_duration = end_t - start_t
            orientation = self.orient_var.get()
            scale_mode = self.scale_mode_var.get()
            pace_choice = self.pace_var.get()
            fps = int(self.fps_var.get())
            clip_audio_setting = self.clip_audio_var.get()
            enable_fade_out = self.fade_out_var.get()
            apply_cinematic_lut = self.fx_cinematic_lut.get()

            # Quality preset settings
            quality_str = self.quality_var.get()
            if "Maximal" in quality_str:
                crf_val = "14"
                preset_val = "veryfast"
            elif "Ultra-Fast" in quality_str:
                crf_val = "20"
                preset_val = "ultrafast"
            else:  # High Quality (Default)
                crf_val = "16"
                preset_val = "veryfast"

            res_choice = self.res_var.get()

            # Spinning Overlay Variables
            has_overlay = self.enable_overlay_var.get() and self.overlay_img_path and os.path.exists(self.overlay_img_path)
            overlay_start_type = self.overlay_start_mode.get()
            overlay_corner_mode = self.overlay_corner_var.get()
            overlay_size_str = self.overlay_size_var.get()
            overlay_speed_val = float(self.slider_ov_speed.get())
            overlay_opacity_val = float(self.slider_ov_opacity.get()) / 100.0
            overlay_circular = self.overlay_circle_cut_var.get()

            # Master audio extraction
            master_audio_path = os.path.join(temp_dir, "master_audio.wav")
            if pre_master_audio and os.path.exists(pre_master_audio):
                shutil.copy(pre_master_audio, master_audio_path)
            else:
                cmd_audio = [
                    self.ffmpeg_exe, "-y",
                    "-ss", f"{start_t:.3f}",
                    "-i", self.audio_path,
                    "-t", f"{total_duration:.3f}",
                    "-vn",
                    "-c:a", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "2",
                    master_audio_path
                ]
                subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if self.is_cancelled:
                raise InterruptedError("Vorgang abgebrochen.")

            # -------------------------------------------------------------
            # STEP 1: Multi-Band Transient Audio-Analyse & Drop-Erkennung
            # -------------------------------------------------------------
            drop_timestamp = 0.0
            if custom_beat_times is None:
                self.set_status("Schritt 1/3: Multi-Band Audio-Analyse & Transient-Tracking...", 0.08)
                sr_target = 22050
                y, sr = librosa.load(master_audio_path, sr=sr_target)
                y_percussive = librosa.effects.percussive(y, margin=2.2)

                # Multi-Band Onset Detection: Kicks (<280Hz) + Snare/Clap Transients (800-5500Hz)
                env_low = librosa.onset.onset_strength(y=y_percussive, sr=sr, fmax=280)
                env_high = librosa.onset.onset_strength(y=y_percussive, sr=sr, fmin=800, fmax=5500)
                onset_env = env_low * 1.0 + env_high * 0.65

                if mode == "onsets":
                    self.set_status("Schritt 1/3: Erfasse dynamische Onset-Peaks...", 0.12)
                    hop_len = 512
                    onset_frames = librosa.onset.onset_detect(
                        onset_envelope=onset_env, sr=sr, hop_length=hop_len,
                        backtrack=True, wait=6, delta=0.07
                    )
                    raw_beats = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_len)
                    beat_times = list(raw_beats)
                else:
                    self.set_status("Schritt 1/3: Berechne Multi-Band Takt & Backtrack-Attack...", 0.12)
                    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                    # CRITICAL: Backtrack beat frames to the exact physical transient attack edge!
                    beat_frames = librosa.onset.onset_backtrack(beat_frames, energy=onset_env)
                    raw_beats = librosa.frames_to_time(beat_frames, sr=sr)

                    rms = librosa.feature.rms(y=y)[0]
                    mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.01

                    # Intelligent Beat-Drop Detection: find maximum sub-bass energy jump after intro (>= 2.5s)
                    diff_low = np.diff(env_low)
                    min_f = int(2.5 * sr / 512)
                    if len(diff_low) > min_f + 10:
                        drop_frame = min_f + int(np.argmax(diff_low[min_f:]))
                        drop_raw_time = float(librosa.frames_to_time(drop_frame, sr=sr))
                        # Snap drop timestamp directly to the nearest beat time
                        nearest_idx = int(np.argmin(np.abs(np.array(raw_beats) - drop_raw_time)))
                        drop_timestamp = float(raw_beats[nearest_idx])
                    else:
                        drop_timestamp = 0.0

                    beat_times = [float(b) for b in raw_beats]
                    if len(beat_times) > 1:
                        merged = []
                        idx = 0
                        while idx < len(beat_times) - 1:
                            t_cur = beat_times[idx]
                            t_next = beat_times[idx + 1]
                            f_start = librosa.time_to_frames(t_cur, sr=sr)
                            f_end = librosa.time_to_frames(t_next, sr=sr)
                            seg_rms = float(np.mean(rms[f_start:f_end])) if f_end > f_start else mean_rms
                            merged.append(t_cur)
                            if seg_rms < mean_rms * 0.60:
                                idx = min(len(beat_times) - 1, idx + 2)
                            else:
                                idx += 1
                        merged.append(beat_times[-1])
                        beat_times = merged

                del y, y_percussive, env_low, env_high, onset_env

                beat_times = sorted([b for b in beat_times if 0.0 < b < total_duration])
                beat_times.insert(0, 0.0)
                beat_times.append(total_duration)

                dense_beats = [beat_times[0]]
                for k in range(len(beat_times) - 1):
                    t1, t2 = beat_times[k], beat_times[k + 1]
                    diff = t2 - t1
                    if diff > 2.5:
                        sub_cuts = int(np.ceil(diff / 1.5))
                        for s in range(1, sub_cuts):
                            dense_beats.append(t1 + (diff * s / sub_cuts))
                    dense_beats.append(t2)
                beat_times = dense_beats

                if "2x Schnell" in pace_choice:
                    fast_beats = []
                    for k in range(len(beat_times) - 1):
                        fast_beats.append(beat_times[k])
                        mid = (beat_times[k] + beat_times[k + 1]) / 2.0
                        if (mid - beat_times[k]) >= 0.15:
                            fast_beats.append(mid)
                    fast_beats.append(beat_times[-1])
                    beat_times = fast_beats
                elif "0.5x Entspannt" in pace_choice:
                    calm_beats = [beat_times[0]]
                    for k in range(1, len(beat_times) - 1, 2):
                        calm_beats.append(beat_times[k])
                    calm_beats.append(beat_times[-1])
                    beat_times = calm_beats

                sanitized_beats = [beat_times[0]]
                for b in beat_times[1:]:
                    if b - sanitized_beats[-1] >= 0.15:
                        sanitized_beats.append(b)
                if sanitized_beats[-1] != total_duration:
                    sanitized_beats[-1] = total_duration
                beat_times = sanitized_beats
            else:
                self.set_status("Schritt 1/3: Verwende manuelle Schnitte...", 0.15)
                beat_times = custom_beat_times
                drop_timestamp = 0.0

            if self.is_cancelled:
                raise InterruptedError("Vorgang abgebrochen.")

            # Frame Quantization (Zero Accumulated Drift)
            raw_frame_indices = [int(round(t * fps)) for t in beat_times]
            clean_frame_indices = [raw_frame_indices[0]]
            min_cut_frames = 5 if fps == 60 else 3
            for fi in raw_frame_indices[1:]:
                if fi - clean_frame_indices[-1] >= min_cut_frames:
                    clean_frame_indices.append(fi)
            total_target_frames = int(round(total_duration * fps))
            clean_frame_indices[-1] = total_target_frames

            frame_counts = [clean_frame_indices[k + 1] - clean_frame_indices[k] for k in range(len(clean_frame_indices) - 1)]
            durations = [fc / float(fps) for fc in frame_counts]

            if has_overlay:
                if overlay_start_type == "drop":
                    active_overlay_start = drop_timestamp
                elif overlay_start_type == "custom":
                    active_overlay_start = max(0.0, float(self.slider_overlay_sec.get()) - start_t)
                else:
                    active_overlay_start = 0.0
            else:
                active_overlay_start = 999999.0

            # -------------------------------------------------------------
            # STEP 2: Video-Zuschnitt & Hardware-Beschleunigte Effekte
            # -------------------------------------------------------------
            self.set_status("Schritt 2/3: Berechne Canvas & Bild-Einpassung...", 0.20)

            orig_w, orig_h = 1920, 1080
            for fp in self.video_paths:
                f_ext = os.path.splitext(fp)[1].lower()
                try:
                    if f_ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                        with PIL.Image.open(fp) as _img:
                            orig_w, orig_h = _img.size
                    else:
                        _probe = VideoFileClip(fp)
                        orig_w, orig_h = _probe.size
                        _probe.close()
                    break
                except Exception:
                    continue

            # Resolution scaling
            if "1440p" in res_choice:
                base_w, base_h = 2560, 1440
            elif "Original" in res_choice:
                base_w, base_h = orig_w, orig_h
            else:
                base_w, base_h = 1920, 1080

            if orientation == "portrait":
                tw, th = min(base_w, base_h), max(base_w, base_h)
            elif orientation == "landscape":
                tw, th = max(base_w, base_h), min(base_w, base_h)
            else:
                tw, th = orig_w, orig_h

            tw = tw - (tw % 2)
            th = th - (th % 2)

            video_positions = {}

            # Pre-build Spinning Token (Dark Triad Style)
            prepared_token = None
            if "Groß (45%" in overlay_size_str:
                token_ratio = 0.45
            elif "Mittel (30%" in overlay_size_str:
                token_ratio = 0.30
            else:
                token_ratio = 0.20

            token_size = int(round(min(tw, th) * token_ratio))
            margin_x = int(round(tw * 0.035))
            margin_y = int(round(th * 0.035))

            center_pos = ((tw - token_size) // 2, (th - token_size) // 2)
            corners = [
                (margin_x, margin_y),                             # Top-Left
                (tw - token_size - margin_x, margin_y),           # Top-Right
                (tw - token_size - margin_x, th - token_size - margin_y), # Bottom-Right
                (margin_x, th - token_size - margin_y)            # Bottom-Left
            ]

            if has_overlay:
                try:
                    raw_ov = PIL.Image.open(self.overlay_img_path).convert("RGBA")
                    if overlay_circular:
                        min_dim = min(raw_ov.width, raw_ov.height)
                        left = (raw_ov.width - min_dim) // 2
                        top = (raw_ov.height - min_dim) // 2
                        sq = raw_ov.crop((left, top, left + min_dim, top + min_dim)).resize((token_size, token_size), PIL.Image.LANCZOS)
                        mask = PIL.Image.new("L", (token_size, token_size), 0)
                        d = PIL.ImageDraw.Draw(mask)
                        d.ellipse((3, 3, token_size - 3, token_size - 3), fill=255)
                        prepared_token = PIL.Image.new("RGBA", (token_size, token_size), (0, 0, 0, 0))
                        prepared_token.paste(sq, (0, 0), mask)
                        d_out = PIL.ImageDraw.Draw(prepared_token)
                        d_out.ellipse((3, 3, token_size - 3, token_size - 3), outline=(255, 255, 255, 220), width=3)
                    else:
                        prepared_token = raw_ov.resize((token_size, token_size), PIL.Image.LANCZOS)
                    raw_ov.close()

                    if overlay_opacity_val < 0.999 and prepared_token is not None:
                        r_ch, g_ch, b_ch, a_ch = prepared_token.split()
                        a_ch = a_ch.point(lambda p: int(p * overlay_opacity_val))
                        prepared_token = PIL.Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))
                except Exception as e:
                    print(f"Overlay preparation error: {e}")
                    prepared_token = None

            def fit_clip(clip, target_w, target_h):
                cw, ch = clip.size
                if scale_mode == "fit":
                    scale = min(target_w / float(cw), target_h / float(ch))
                    resized = clip.resize(scale)
                    return resized.on_color(size=(target_w, target_h), color=(0, 0, 0), pos="center")
                else:
                    scale = max(target_w / float(cw), target_h / float(ch)) * 1.002
                    resized = clip.resize(scale)
                    rw, rh = resized.size
                    return resized.crop(
                        x_center=rw / 2.0,
                        y_center=rh / 2.0,
                        width=target_w,
                        height=target_h,
                    )

            # In-Memory Cache for Image Clips
            def open_source(path):
                ext_ = os.path.splitext(path)[1].lower()
                if ext_ in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                    if path in self.source_image_cache:
                        return ImageClip(self.source_image_cache[path])
                    img_ = PIL.Image.open(path).convert("RGB")
                    if img_.width > tw * 1.5 or img_.height > th * 1.5:
                        scale = max(tw / img_.width, th / img_.height)
                        nw, nh = max(1, int(img_.width * scale)), max(1, int(img_.height * scale))
                        img_ = img_.resize((nw, nh), resample=PIL.Image.BILINEAR)
                    arr_ = np.array(img_)
                    self.source_image_cache[path] = arr_
                    img_.close()
                    return ImageClip(arr_)
                else:
                    return VideoFileClip(path)

            def cut_source(src, dur, path=None):
                if isinstance(src, ImageClip):
                    return src.set_duration(dur)
                safe_max_dur = max(0.1, src.duration - 0.05)
                if path is not None and path in video_positions:
                    cs = video_positions[path]
                    if cs >= safe_max_dur:
                        cs = 0.0
                else:
                    max_s = max(0, safe_max_dur - dur)
                    cs = random.uniform(0, max_s) if max_s > 0 else 0.0
                end_s = min(cs + dur, safe_max_dur)
                sc = src.subclip(cs, end_s)
                if path is not None:
                    next_pos = end_s if end_s < safe_max_dur else 0.0
                    video_positions[path] = next_pos
                if sc.duration < dur:
                    sc = sc.fx(vfx.loop, duration=dur)
                return sc

            # =========================================================
            # OpenCV Hardware-Accelerated Modern Effect Suite
            # =========================================================
            # 1. Warm Studio Bloom Flash (Smooth Cubic Ease-Out)
            def fx_bloom_flash(clip, dur):
                fd = min(0.14, dur)
                def f(gf, t):
                    frame = gf(t)
                    if t < fd:
                        alpha = 0.58 * ((1.0 - (t / fd)) ** 3)
                        # Warm golden-white tint [255, 252, 240]
                        blend = frame.astype(np.float32) * (1.0 - alpha) + np.array([255, 252, 240], dtype=np.float32) * alpha
                        return np.clip(blend, 0, 255).astype(np.uint8)
                    return frame
                return clip.fl(f)

            # 2. Camera Shake with OpenCV Reflection (Zero Black Borders!)
            def fx_camera_shake(clip, dur):
                sd = min(0.18, dur)
                def f(gf, t):
                    frame = gf(t)
                    if t < sd:
                        decay = (1.0 - (t / sd)) ** 3
                        dx = np.sin(t * 45.0) * 14.0 * decay
                        dy = np.cos(t * 35.0) * 9.0 * decay
                        M = np.float32([[1, 0, dx], [0, 1, dy]])
                        return cv2.warpAffine(frame, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    return frame
                return clip.fl(f)

            # 3. Optical Punch Zoom with OpenCV Ease-Out
            def fx_punch_zoom(clip, dur, p=1.06):
                def f(gf, t):
                    decay = (1.0 - (t / dur)) ** 3
                    s = 1.0 + (p - 1.0) * decay
                    frame = gf(t)
                    if s > 1.002:
                        M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), 0, s)
                        return cv2.warpAffine(frame, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    return frame
                return clip.fl(f)

            # 4. Cinematic Slow Push-In
            def fx_slow_push(clip, dur):
                def f(gf, t):
                    s = 1.0 + (0.035 * (t / dur))
                    frame = gf(t)
                    if s > 1.002:
                        M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), 0, s)
                        return cv2.warpAffine(frame, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    return frame
                return clip.fl(f)

            # 5. Transient RGB Shockwave
            def fx_rgb_shockwave(clip, dur):
                sd = min(0.07, dur)
                def f(gf, t):
                    frame = gf(t).copy()
                    if t < sd:
                        shift = int(round(6.0 * (1.0 - t / sd)))
                        if shift > 0:
                            frame[:, :, 0] = np.roll(frame[:, :, 0], shift, axis=1)
                            frame[:, :, 2] = np.roll(frame[:, :, 2], -shift, axis=1)
                    return frame
                return clip.fl(f)

            # 6. Cinematic Color Grading LUT
            def apply_color_lut(clip):
                return clip.fl_image(lambda img: cv2.LUT(img, CINEMATIC_LUT))

            # Dark Triad Spinning Overlay Compositor
            def apply_spinning_overlay(clip, cur_cut_idx, cut_global_start, dur):
                if prepared_token is None:
                    return clip

                if "In der Mitte" in overlay_corner_mode:
                    target_pos = center_pos
                    prev_pos = center_pos
                elif "Wechselt zum Takt" in overlay_corner_mode:
                    target_corner_idx = cur_cut_idx % 4
                    prev_corner_idx = (cur_cut_idx - 1) % 4
                    target_pos = corners[target_corner_idx]
                    prev_pos = corners[prev_corner_idx]
                elif "Oben Links" in overlay_corner_mode:
                    target_pos = corners[0]
                    prev_pos = corners[0]
                elif "Oben Rechts" in overlay_corner_mode:
                    target_pos = corners[1]
                    prev_pos = corners[1]
                elif "Unten Rechts" in overlay_corner_mode:
                    target_pos = corners[2]
                    prev_pos = corners[2]
                else:  # Unten Links
                    target_pos = corners[3]
                    prev_pos = corners[3]

                def overlay_frame(gf, t):
                    frame = gf(t)
                    global_t = cut_global_start + t

                    slide_dur = 0.12
                    if "Wechselt zum Takt" in overlay_corner_mode and t < slide_dur and cur_cut_idx > 0:
                        prog = min(1.0, t / slide_dur)
                        ease = 1.0 - ((1.0 - prog) ** 3)
                        cx = int(prev_pos[0] + (target_pos[0] - prev_pos[0]) * ease)
                        cy = int(prev_pos[1] + (target_pos[1] - prev_pos[1]) * ease)
                    else:
                        cx, cy = target_pos

                    angle = (global_t * 360.0 * overlay_speed_val) % 360.0
                    rotated = prepared_token.rotate(-angle, resample=PIL.Image.BILINEAR)

                    bg_pil = PIL.Image.fromarray(frame.astype(np.uint8))
                    bg_pil.paste(rotated, (cx, cy), rotated)
                    return np.array(bg_pil)

                return clip.fl(overlay_frame)

            allow_shake = self.fx_shake.get()
            allow_bloom = self.fx_bloom_flash.get()
            allow_punch = self.fx_punch_zoom.get()
            allow_push = self.fx_slow_push.get()
            allow_rgb = self.fx_rgb_shock.get()
            allow_speedramp = self.fx_speedramp.get()

            if "Dezent 15%" in clip_audio_setting:
                clip_audio_vol = 0.15
            elif "Mittel 35%" in clip_audio_setting:
                clip_audio_vol = 0.35
            else:
                clip_audio_vol = 0.0

            cpu_threads = max(4, min(16, os.cpu_count() or 4))
            video_paths = self.video_paths
            clip_idx = 0
            i = 0
            total_dur_count = len(durations)

            while i < total_dur_count:
                if self.is_cancelled:
                    raise InterruptedError("Vorgang durch Benutzer abgebrochen.")

                duration = float(durations[i])
                if duration < 0.10:
                    duration = 0.10

                cut_global_start = clean_frame_indices[i] / float(fps)
                cur_path = video_paths[clip_idx]
                source_clip = open_source(cur_path)

                # Modern Effect Selection
                possible_fx = ["clean"]
                if allow_bloom: possible_fx.append("bloom")
                if allow_shake: possible_fx.append("shake")
                if allow_punch: possible_fx.append("punch")
                if allow_push and duration >= 0.5: possible_fx.append("push")
                if allow_rgb: possible_fx.append("rgb")
                if allow_speedramp and not isinstance(source_clip, ImageClip): possible_fx.append("speedramp")

                chosen_fx = random.choice(possible_fx)

                if chosen_fx == "speedramp":
                    speed_mult = 1.30
                    raw_cut = cut_source(source_clip, duration * speed_mult, cur_path)
                    sub_clip = fit_clip(raw_cut, tw, th).fx(vfx.speedx, speed_mult)
                    if sub_clip.duration < duration:
                        sub_clip = sub_clip.fx(vfx.loop, duration=duration)
                    else:
                        sub_clip = sub_clip.subclip(0, duration)
                else:
                    raw_cut = cut_source(source_clip, duration, cur_path)
                    sub_clip = fit_clip(raw_cut, tw, th)

                    if chosen_fx == "bloom":
                        sub_clip = fx_bloom_flash(sub_clip, duration)
                    elif chosen_fx == "shake":
                        sub_clip = fx_camera_shake(sub_clip, duration)
                    elif chosen_fx == "punch":
                        sub_clip = fx_punch_zoom(sub_clip, duration, 1.06)
                    elif chosen_fx == "push":
                        sub_clip = fx_slow_push(sub_clip, duration)
                    elif chosen_fx == "rgb":
                        sub_clip = fx_rgb_shockwave(sub_clip, duration)

                # Apply Cinematic Color Grade LUT if active
                if apply_cinematic_lut:
                    sub_clip = apply_color_lut(sub_clip)

                # Apply Spinning Overlay if drop threshold is reached
                if has_overlay and cut_global_start >= active_overlay_start:
                    sub_clip = apply_spinning_overlay(sub_clip, i, cut_global_start, duration)

                # Safe audio handling
                seg_has_audio = False
                if clip_audio_vol > 0.0:
                    try:
                        if raw_cut.audio is not None:
                            sub_clip = sub_clip.set_audio(raw_cut.audio.volumex(clip_audio_vol))
                            seg_has_audio = True
                    except Exception:
                        seg_has_audio = False

                if not seg_has_audio:
                    sub_clip = sub_clip.without_audio()

                temp_path = os.path.join(temp_dir, f"seg_{len(temp_files):04d}.mp4")
                write_args = {
                    "fps": fps,
                    "codec": "libx264",
                    "threads": cpu_threads,
                    "logger": None,
                    "ffmpeg_params": [
                        "-preset", preset_val,
                        "-crf", crf_val,
                        "-tune", "film",
                        "-pix_fmt", "yuv420p"
                    ]
                }
                if seg_has_audio:
                    write_args["audio_codec"] = "aac"

                sub_clip.write_videofile(temp_path, **write_args)
                temp_files.append(temp_path)

                source_clip.close()
                try:
                    sub_clip.close()
                except Exception:
                    pass

                clip_idx = (clip_idx + 1) % len(video_paths)
                i += 1

                pct = 0.20 + (0.68 * (i / total_dur_count))
                self.set_status(f"Schritt 2/3: Erstelle Beat-Clip {len(temp_files)}/{total_dur_count} ({fps} FPS, {tw}x{th})...", pct)

            if self.is_cancelled:
                raise InterruptedError("Vorgang durch Benutzer abgebrochen.")

            # -------------------------------------------------------------
            # STEP 3: Ultra-Fast Concat + Studio Master Audio Mux (320 kbps)
            # -------------------------------------------------------------
            self.set_status("Schritt 3/3: Finale Studio-Zusammensetzung & Audio-Muxing...", 0.90)

            concat_txt = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_txt, "w", encoding="utf-8") as f:
                for tp in temp_files:
                    safe_p = tp.replace("\\", "/")
                    f.write(f"file '{safe_p}'\n")

            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_txt,
                "-i", master_audio_path,
            ]

            fade_duration = 0.8
            fade_start = max(0.0, total_duration - fade_duration)

            if clip_audio_vol > 0.0:
                af_filter = f"[0:a]volume=1.0[a0];[1:a]volume=1.0[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2"
                if enable_fade_out:
                    af_filter += f",afade=t=out:st={fade_start:.3f}:d={fade_duration:.3f}[aout]"
                else:
                    af_filter += "[aout]"

                cmd.extend([
                    "-filter_complex", af_filter,
                    "-map", "0:v:0",
                    "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "320k",
                    "-shortest",
                    out_path
                ])
            else:
                if enable_fade_out:
                    cmd.extend([
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "320k",
                        "-af", f"afade=t=out:st={fade_start:.3f}:d={fade_duration:.3f}",
                        "-shortest",
                        out_path
                    ])
                else:
                    cmd.extend([
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "320k",
                        "-shortest",
                        out_path
                    ])

            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg Concat Fehler (Code {process.returncode}):\n{process.stderr}")

            self.last_rendered_path = out_path
            self.set_status("Fertig! Edit erfolgreich gerendert.", 1.0)
            self.after(0, lambda: self.post_action_frame.pack(fill="x", padx=20, pady=(4, 10)))
            self.show_info("Erfolg", f"Dein Edit ({fps} FPS, {tw}x{th}) wurde in Studio-Qualität perfekt synchron gespeichert unter:\n{out_path}")

        except InterruptedError:
            self.set_status("Rendern abgebrochen.", 0.0)
            self.show_info("Abgebrochen", "Der Rendervorgang wurde sauber beendet.")
        except Exception as e:
            err = traceback.format_exc()
            print(err)
            self.set_status("Fehler beim Rendern aufgetreten!", 0.0)
            self.show_error("Fehler", f"Ein Fehler ist aufgetreten:\n{e}")
        finally:
            self.btn_generate.configure(state="normal")
            self.btn_cancel.configure(state="disabled")
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    app = EditMakerApp()
    app.mainloop()
