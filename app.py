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
import json
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

def _build_vignette_mask(w, h):
    X = cv2.getGaussianKernel(w, w * 0.48)
    Y = cv2.getGaussianKernel(h, h * 0.48)
    kernel = Y * X.T
    kernel = kernel / kernel.max()
    vig = np.clip(0.68 + 0.32 * kernel, 0, 1)
    vig_u8 = np.clip(vig * 255.0, 0, 255).astype(np.uint8)
    return cv2.merge([vig_u8, vig_u8, vig_u8])

# Default Presets
DEFAULT_PRESETS = {
    "TikTok Montage / Beat Drop": {
        "mode": "auto",
        "pace": "1x (Jeder Takt)",
        "orientation": "portrait",
        "scale_mode": "fill",
        "fps": 60,
        "quality": "High Quality (CRF 16 Studio)",
        "resolution": "1080p (Full HD)",
        "clip_audio": "Stumm (100% Musik)",
        "fade_out": True,
        "combo_style": "Impact-Combo (Gleichzeitig auf Beat-Hits gestapelt)",
        "fx_intensity": 130,
        "fx_shake": True,
        "fx_bloom_flash": True,
        "fx_punch_zoom": True,
        "fx_slow_push": True,
        "fx_rgb_shock": True,
        "fx_vignette": True,
        "fx_speedramp": True,
        "fx_cinematic_lut": True,
        "enable_overlay": False,
        "overlay_corner": "In der Mitte (Zentriert)",
        "overlay_size": "Gross (45% Bildhoehe)",
        "overlay_speed": 1.8,
        "overlay_opacity": 100,
        "overlay_start_mode": "drop",
        "overlay_circle": True
    },
    "Velocity / Action Edit": {
        "mode": "onsets",
        "pace": "2x Schnell (Halber Takt)",
        "orientation": "portrait",
        "scale_mode": "fill",
        "fps": 60,
        "quality": "High Quality (CRF 16 Studio)",
        "resolution": "1080p (Full HD)",
        "clip_audio": "Dezent 15% (Killsounds/SFX)",
        "fade_out": True,
        "combo_style": "Impact-Combo (Gleichzeitig auf Beat-Hits gestapelt)",
        "fx_intensity": 100,
        "fx_shake": True,
        "fx_bloom_flash": True,
        "fx_punch_zoom": True,
        "fx_slow_push": False,
        "fx_rgb_shock": True,
        "fx_vignette": True,
        "fx_speedramp": True,
        "fx_cinematic_lut": True,
        "enable_overlay": False,
        "overlay_corner": "Wechselt zum Takt (Animiert)",
        "overlay_size": "Mittel (30% Bildhoehe)",
        "overlay_speed": 1.2,
        "overlay_opacity": 100,
        "overlay_start_mode": "drop",
        "overlay_circle": True
    },
    "Cinematic Flow & Aesthetic": {
        "mode": "auto",
        "pace": "1x (Jeder Takt)",
        "orientation": "landscape",
        "scale_mode": "fill",
        "fps": 60,
        "quality": "Maximal (CRF 14 Glasklar)",
        "resolution": "1440p (2K Ultra Sharp)",
        "clip_audio": "Stumm (100% Musik)",
        "fade_out": True,
        "combo_style": "Dezent & Sanft (Nur weiche Uebergaenge)",
        "fx_intensity": 70,
        "fx_shake": False,
        "fx_bloom_flash": True,
        "fx_punch_zoom": False,
        "fx_slow_push": True,
        "fx_rgb_shock": False,
        "fx_vignette": True,
        "fx_speedramp": False,
        "fx_cinematic_lut": True,
        "enable_overlay": False,
        "overlay_corner": "Oben Rechts",
        "overlay_size": "Kompakt (20% Bildhoehe)",
        "overlay_speed": 1.0,
        "overlay_opacity": 85,
        "overlay_start_mode": "start",
        "overlay_circle": True
    },
    "Clean & Minimal (Pure Cuts)": {
        "mode": "auto",
        "pace": "1x (Jeder Takt)",
        "orientation": "portrait",
        "scale_mode": "fill",
        "fps": 60,
        "quality": "High Quality (CRF 16 Studio)",
        "resolution": "1080p (Full HD)",
        "clip_audio": "Stumm (100% Musik)",
        "fade_out": True,
        "combo_style": "Dezent & Sanft (Nur weiche Uebergaenge)",
        "fx_intensity": 40,
        "fx_shake": False,
        "fx_bloom_flash": False,
        "fx_punch_zoom": False,
        "fx_slow_push": False,
        "fx_rgb_shock": False,
        "fx_vignette": False,
        "fx_speedramp": False,
        "fx_cinematic_lut": False,
        "enable_overlay": False,
        "overlay_corner": "In der Mitte (Zentriert)",
        "overlay_size": "Mittel (30% Bildhoehe)",
        "overlay_speed": 1.0,
        "overlay_opacity": 100,
        "overlay_start_mode": "start",
        "overlay_circle": True
    }
}

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

        self.presets_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.json")
        self.all_presets = self._load_presets_from_disk()

        pygame.mixer.init()

        self._build_ui()

    def _load_presets_from_disk(self):
        presets = dict(DEFAULT_PRESETS)
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        presets.update(data)
            except Exception as e:
                print(f"Fehler beim Laden von presets.json: {e}")
        else:
            try:
                with open(self.presets_file, "w", encoding="utf-8") as f:
                    json.dump(presets, f, indent=4, ensure_ascii=False)
            except Exception:
                pass
        return presets

    def _save_presets_to_disk(self):
        try:
            with open(self.presets_file, "w", encoding="utf-8") as f:
                json.dump(self.all_presets, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Fehler beim Speichern von presets.json: {e}")

    # =====================================================================
    # UI Building
    # =====================================================================
    def _build_ui(self):
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=15, pady=12)

        # 1. Header
        lbl_title = ctk.CTkLabel(
            self.main_container,
            text="Auto Beat-Sync Edit Maker",
            font=("Arial", 26, "bold")
        )
        lbl_title.pack(pady=(4, 2))
        
        lbl_sub = ctk.CTkLabel(
            self.main_container,
            text="Pro Studio Engine | Multi-Band Transient Sync | 60 FPS | OpenCV Hardware-Beschleunigung",
            text_color="#38bdf8",
            font=("Arial", 12)
        )
        lbl_sub.pack(pady=(0, 10))

        # Profile & Preset Management Frame
        profile_frame = ctk.CTkFrame(self.main_container)
        profile_frame.pack(fill="x", pady=5)

        header_prof = ctk.CTkFrame(profile_frame, fg_color="transparent")
        header_prof.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(header_prof, text="Profile & Presets (Speichern & Laden)", font=("Arial", 14, "bold")).pack(side="left")

        row_prof = ctk.CTkFrame(profile_frame, fg_color="transparent")
        row_prof.pack(fill="x", padx=12, pady=(3, 8))

        ctk.CTkLabel(row_prof, text="Profil:").pack(side="left", padx=(0, 6))
        self.preset_names = list(self.all_presets.keys())
        self.profile_var = ctk.StringVar(value=self.preset_names[0] if self.preset_names else "Standard")
        self.combo_profile = ctk.CTkComboBox(
            row_prof,
            values=self.preset_names,
            variable=self.profile_var,
            command=self.on_profile_selected,
            width=230
        )
        self.combo_profile.pack(side="left", padx=4)

        self.btn_save_profile = ctk.CTkButton(
            row_prof,
            text="Profil speichern",
            command=self.save_current_as_profile,
            fg_color="#1f8a4c",
            hover_color="#176b3b",
            width=120
        )
        self.btn_save_profile.pack(side="left", padx=4)

        self.btn_load_json = ctk.CTkButton(
            row_prof,
            text="JSON laden",
            command=self.import_profile_json,
            fg_color="#2b719e",
            hover_color="#1f5375",
            width=90
        )
        self.btn_load_json.pack(side="left", padx=4)

        self.btn_export_json = ctk.CTkButton(
            row_prof,
            text="Export",
            command=self.export_profile_json,
            fg_color="#444444",
            hover_color="#555555",
            width=75
        )
        self.btn_export_json.pack(side="left", padx=4)

        self.btn_delete_prof = ctk.CTkButton(
            row_prof,
            text="Loeschen",
            command=self.delete_current_profile,
            fg_color="#8b1e1e",
            hover_color="#631414",
            width=75
        )
        self.btn_delete_prof.pack(side="left", padx=4)

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
            text="Clips / Bilder wählen",
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
            text="Song auswählen",
            command=self.select_audio,
            width=180
        )
        self.btn_select_audio.pack(side="left", padx=(0, 10))

        self.btn_audio_preview = ctk.CTkButton(
            btn_row_audio,
            text="Vorhören",
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

        # 4. Beat Detection & Pacing Frame (Multi-Band Engine)
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
        ctk.CTkRadioButton(row_fmt, text="Hochkant (9:16)", variable=self.orient_var, value="portrait").pack(side="left", padx=6)
        ctk.CTkRadioButton(row_fmt, text="Quer (16:9)", variable=self.orient_var, value="landscape").pack(side="left", padx=6)

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
        self.quality_var = ctk.StringVar(value="High Quality (CRF 16 Studio)")
        ctk.CTkComboBox(
            row_quality,
            values=[
                "High Quality (CRF 16 Studio)",
                "Maximal (CRF 14 Glasklar)",
                "Ultra-Fast (CRF 20 Schnell)"
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

        ctk.CTkLabel(preset_row, text="Effekt-Stil:").pack(side="left", padx=(0, 8))
        self.combo_fx_style = ctk.StringVar(value="Impact-Combo (Gleichzeitig auf Beat-Hits gestapelt)")
        ctk.CTkComboBox(
            preset_row,
            values=[
                "Impact-Combo (Gleichzeitig auf Beat-Hits gestapelt)",
                "Zufällig wechselnd (Ein Effekt pro Cut)",
                "Dezent & Sanft (Nur weiche Übergänge)"
            ],
            variable=self.combo_fx_style,
            width=390
        ).pack(side="left", padx=5)

        self.fx_cinematic_lut = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            preset_row,
            text="Cinematic S-Curve Contrast & Color Pop",
            variable=self.fx_cinematic_lut
        ).pack(side="left", padx=(15, 5))

        # Intensity Slider Row
        intensity_row = ctk.CTkFrame(effects_frame, fg_color="transparent")
        intensity_row.pack(fill="x", padx=12, pady=(4, 2))

        self.lbl_fx_intensity = ctk.CTkLabel(intensity_row, text="Effekt-Intensität: 100% (Recoil & Bloom)", width=280, anchor="w")
        self.lbl_fx_intensity.pack(side="left", padx=(0, 10))

        self.slider_fx_intensity = ctk.CTkSlider(
            intensity_row,
            from_=20, to=200,
            command=lambda v: self.lbl_fx_intensity.configure(text=f"Effekt-Intensität: {int(v)}% (Recoil & Bloom)")
        )
        self.slider_fx_intensity.set(100)
        self.slider_fx_intensity.pack(side="left", fill="x", expand=True, padx=5)

        eff_grid = ctk.CTkFrame(effects_frame, fg_color="transparent")
        eff_grid.pack(fill="x", padx=12, pady=(6, 10))

        self.fx_shake = ctk.BooleanVar(value=True)
        self.fx_bloom_flash = ctk.BooleanVar(value=True)
        self.fx_punch_zoom = ctk.BooleanVar(value=True)
        self.fx_slow_push = ctk.BooleanVar(value=True)
        self.fx_rgb_shock = ctk.BooleanVar(value=True)
        self.fx_vignette = ctk.BooleanVar(value=True)
        self.fx_speedramp = ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(eff_grid, text="Bass-Shake (Rotations-Tilt & Recoil)", variable=self.fx_shake).grid(row=0, column=0, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Exposure Bloom-Flash (Highlight-Burst)", variable=self.fx_bloom_flash).grid(row=0, column=1, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Punch-Zoom (Optischer Ease)", variable=self.fx_punch_zoom).grid(row=0, column=2, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Radial RGB-Aberration (Linsenzerlegung)", variable=self.fx_rgb_shock).grid(row=1, column=0, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Cinematic Vignette (Dunkler Randfokus)", variable=self.fx_vignette).grid(row=1, column=1, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Slow Push-In (Sanfte Kamerafahrt)", variable=self.fx_slow_push).grid(row=1, column=2, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Speed-Ramp (Sync-Safe Velocity)", variable=self.fx_speedramp).grid(row=2, column=0, sticky="w", padx=8, pady=3)

        # 8. Spinning Overlay Frame
        overlay_frame = ctk.CTkFrame(self.main_container)
        overlay_frame.pack(fill="x", pady=5)

        overlay_header = ctk.CTkFrame(overlay_frame, fg_color="transparent")
        overlay_header.pack(fill="x", padx=12, pady=(10, 4))

        self.enable_overlay_var = ctk.BooleanVar(value=False)
        self.chk_overlay = ctk.CTkCheckBox(
            overlay_header,
            text="Spinning Bild-Overlay aktivieren",
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
            text="Overlay-Bild wählen (PNG/JPG)",
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
        ctk.CTkRadioButton(row_ov2, text="Ab Beat-Drop (Automatisch)", variable=self.overlay_start_mode, value="drop").pack(side="left", padx=6)
        ctk.CTkRadioButton(row_ov2, text="Ab bestimmter Zeit", variable=self.overlay_start_mode, value="custom").pack(side="left", padx=6)
        ctk.CTkRadioButton(row_ov2, text="Von Anfang an", variable=self.overlay_start_mode, value="start").pack(side="left", padx=6)

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
                "In der Mitte (Zentriert)",
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
            text="Video Rendern (Pro Studio Engine)",
            command=self.start_generation,
            fg_color="#1f8a4c",
            hover_color="#176b3b",
            height=44,
            font=("Arial", 16, "bold")
        )
        self.btn_generate.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_cancel = ctk.CTkButton(
            btn_row_action,
            text="Abbrechen",
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
            text="Ordner im Explorer öffnen",
            command=self.open_output_folder,
            fg_color="#2b719e",
            hover_color="#1f5375",
            width=200
        )
        self.btn_open_folder.pack(side="left", padx=5)

        self.btn_play_video = ctk.CTkButton(
            self.post_action_frame,
            text="Video abspielen",
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
    # Preset & Profile Management
    # =====================================================================
    def on_profile_selected(self, choice):
        if choice in self.all_presets:
            self.apply_preset_dict(self.all_presets[choice])

    def get_current_settings_dict(self):
        return {
            "mode": self.mode_var.get(),
            "pace": self.pace_var.get(),
            "orientation": self.orient_var.get(),
            "scale_mode": self.scale_mode_var.get(),
            "fps": int(self.fps_var.get()),
            "quality": self.quality_var.get(),
            "resolution": self.res_var.get(),
            "clip_audio": self.clip_audio_var.get(),
            "fade_out": self.fade_out_var.get(),
            "combo_style": self.combo_fx_style.get(),
            "fx_intensity": int(self.slider_fx_intensity.get()),
            "fx_shake": self.fx_shake.get(),
            "fx_bloom_flash": self.fx_bloom_flash.get(),
            "fx_punch_zoom": self.fx_punch_zoom.get(),
            "fx_slow_push": self.fx_slow_push.get(),
            "fx_rgb_shock": self.fx_rgb_shock.get(),
            "fx_vignette": self.fx_vignette.get(),
            "fx_speedramp": self.fx_speedramp.get(),
            "fx_cinematic_lut": self.fx_cinematic_lut.get(),
            "enable_overlay": self.enable_overlay_var.get(),
            "overlay_corner": self.overlay_corner_var.get(),
            "overlay_size": self.overlay_size_var.get(),
            "overlay_speed": float(self.slider_ov_speed.get()),
            "overlay_opacity": float(self.slider_ov_opacity.get()),
            "overlay_start_mode": self.overlay_start_mode.get(),
            "overlay_circle": self.overlay_circle_cut_var.get()
        }

    def apply_preset_dict(self, data):
        if not isinstance(data, dict):
            return
        if "mode" in data: self.mode_var.set(data["mode"])
        if "pace" in data: self.pace_var.set(data["pace"])
        if "orientation" in data: self.orient_var.set(data["orientation"])
        if "scale_mode" in data: self.scale_mode_var.set(data["scale_mode"])
        if "fps" in data: self.fps_var.set(int(data["fps"]))
        if "quality" in data: self.quality_var.set(data["quality"])
        if "resolution" in data: self.res_var.set(data["resolution"])
        if "clip_audio" in data: self.clip_audio_var.set(data["clip_audio"])
        if "fade_out" in data: self.fade_out_var.set(bool(data["fade_out"]))
        if "combo_style" in data: self.combo_fx_style.set(data["combo_style"])
        if "fx_intensity" in data:
            val = float(data["fx_intensity"])
            self.slider_fx_intensity.set(val)
            self.lbl_fx_intensity.configure(text=f"Effekt-Intensität: {int(val)}% (Recoil & Bloom)")
        if "fx_shake" in data: self.fx_shake.set(bool(data["fx_shake"]))
        if "fx_bloom_flash" in data: self.fx_bloom_flash.set(bool(data["fx_bloom_flash"]))
        if "fx_punch_zoom" in data: self.fx_punch_zoom.set(bool(data["fx_punch_zoom"]))
        if "fx_slow_push" in data: self.fx_slow_push.set(bool(data["fx_slow_push"]))
        if "fx_rgb_shock" in data: self.fx_rgb_shock.set(bool(data["fx_rgb_shock"]))
        if "fx_vignette" in data: self.fx_vignette.set(bool(data["fx_vignette"]))
        if "fx_speedramp" in data: self.fx_speedramp.set(bool(data["fx_speedramp"]))
        if "fx_cinematic_lut" in data: self.fx_cinematic_lut.set(bool(data["fx_cinematic_lut"]))
        if "enable_overlay" in data:
            self.enable_overlay_var.set(bool(data["enable_overlay"]))
            self._toggle_overlay_ui()
        if "overlay_corner" in data: self.overlay_corner_var.set(data["overlay_corner"])
        if "overlay_size" in data: self.overlay_size_var.set(data["overlay_size"])
        if "overlay_speed" in data:
            spd = float(data["overlay_speed"])
            self.slider_ov_speed.set(spd)
            self.lbl_ov_speed.configure(text=f"Dreh-Speed: {spd:.1f}x (Umdr./s)")
        if "overlay_opacity" in data:
            op = float(data["overlay_opacity"])
            self.slider_ov_opacity.set(op)
            self.lbl_ov_opacity.configure(text=f"Deckkraft: {int(op)}%")
        if "overlay_start_mode" in data: self.overlay_start_mode.set(data["overlay_start_mode"])
        if "overlay_circle" in data: self.overlay_circle_cut_var.set(bool(data["overlay_circle"]))

    def save_current_as_profile(self):
        dialog = ctk.CTkInputDialog(title="Profil speichern", text="Name für das neue Profil:")
        name = dialog.get_input()
        if name and name.strip():
            name = name.strip()
            self.all_presets[name] = self.get_current_settings_dict()
            self._save_presets_to_disk()
            values = list(self.all_presets.keys())
            self.combo_profile.configure(values=values)
            self.profile_var.set(name)
            self.set_status(f"Profil '{name}' erfolgreich gespeichert.", 0.0)

    def import_profile_json(self):
        path = filedialog.askopenfilename(
            title="Profil aus JSON-Datei laden",
            filetypes=[("JSON-Dateien", "*.json"), ("Alle Dateien", "*.*")]
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "mode" in data or "fps" in data:
                        base_name = os.path.splitext(os.path.basename(path))[0]
                        self.all_presets[base_name] = data
                        self._save_presets_to_disk()
                        self.combo_profile.configure(values=list(self.all_presets.keys()))
                        self.profile_var.set(base_name)
                        self.apply_preset_dict(data)
                        self.set_status(f"Profil aus '{os.path.basename(path)}' geladen.", 0.0)
                    else:
                        self.all_presets.update(data)
                        self._save_presets_to_disk()
                        self.combo_profile.configure(values=list(self.all_presets.keys()))
                        self.set_status("Preset-Bundle importiert.", 0.0)
            except Exception as e:
                self.show_error("Fehler", f"Konnte Profil nicht laden:\n{e}")

    def export_profile_json(self):
        path = filedialog.asksaveasfilename(
            title="Aktuelles Profil exportieren",
            defaultextension=".json",
            filetypes=[("JSON-Dateien", "*.json")]
        )
        if path:
            try:
                current_data = self.get_current_settings_dict()
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(current_data, f, indent=4, ensure_ascii=False)
                self.set_status(f"Profil nach '{os.path.basename(path)}' exportiert.", 0.0)
            except Exception as e:
                self.show_error("Fehler", f"Export fehlgeschlagen:\n{e}")

    def delete_current_profile(self):
        cur = self.profile_var.get()
        if cur in DEFAULT_PRESETS:
            messagebox.showinfo("Hinweis", f"Das Standard-Profil '{cur}' kann nicht gelöscht werden.")
            return
        if cur in self.all_presets:
            del self.all_presets[cur]
            self._save_presets_to_disk()
            values = list(self.all_presets.keys())
            self.combo_profile.configure(values=values)
            new_sel = values[0] if values else "Standard"
            self.profile_var.set(new_sel)
            if new_sel in self.all_presets:
                self.apply_preset_dict(self.all_presets[new_sel])
            self.set_status(f"Profil '{cur}' gelöscht.", 0.0)

    # =====================================================================
    # Thread-Safe UI Update Helpers
    # =====================================================================
    def set_status(self, text, progress=None):
        def _update():
            self.lbl_status.configure(text=text)
            if progress is not None:
                p_clamped = max(0.0, min(1.0, progress))
                self.progress_bar.set(p_clamped)
                self.lbl_pct.configure(text=f"{int(p_clamped * 100)}%")
        self.after(0, _update)

    def show_error(self, title, msg):
        self.after(0, lambda: messagebox.showerror(title, msg))

    def show_info(self, title, msg):
        self.after(0, lambda: messagebox.showinfo(title, msg))

    # =====================================================================
    # Media Selection & Preview
    # =====================================================================
    def select_videos(self):
        paths = filedialog.askopenfilenames(
            title="Wähle Videos oder Bilder aus",
            filetypes=[
                ("Medien-Dateien", "*.mp4 *.mov *.avi *.mkv *.png *.jpg *.jpeg *.webp *.bmp"),
                ("Video-Dateien", "*.mp4 *.mov *.avi *.mkv"),
                ("Bilder", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("Alle Dateien", "*.*")
            ]
        )
        if paths:
            new_paths = [p for p in paths if p not in self.video_paths]
            self.video_paths.extend(new_paths)
            self._update_media_listbox()

    def clear_videos(self):
        self.video_paths = []
        self._update_media_listbox()

    def _update_media_listbox(self):
        self.media_listbox.configure(state="normal")
        self.media_listbox.delete("1.0", "end")
        count = len(self.video_paths)
        if count == 0:
            self.media_listbox.insert("1.0", "Noch keine Dateien gewählt. Unterstützt MP4, MOV, MKV, AVI, PNG, JPG, WEBP...")
            self.lbl_videos_count.configure(text="0 Dateien ausgewählt", text_color="gray")
        else:
            names = [f"[{i+1}] {os.path.basename(p)}" for i, p in enumerate(self.video_paths)]
            self.media_listbox.insert("1.0", "\n".join(names))
            self.lbl_videos_count.configure(text=f"{count} Dateien bereit", text_color="#38bdf8")
        self.media_listbox.configure(state="disabled")

    def select_audio(self):
        path = filedialog.askopenfilename(
            title="Wähle eine Musik-Datei",
            filetypes=[("Audio-Dateien", "*.mp3 *.wav *.ogg *.aac *.flac *.m4a"), ("Alle Dateien", "*.*")]
        )
        if path:
            self.audio_path = path
            try:
                probe = AudioFileClip(path)
                self.audio_duration = float(probe.duration)
                probe.close()
            except Exception:
                try:
                    self.audio_duration = float(librosa.get_duration(path=path))
                except Exception as e:
                    self.show_error("Fehler", f"Konnte Audio-Dauer nicht ermitteln:\n{e}")
                    return

            self.slider_start.configure(to=self.audio_duration)
            self.slider_end.configure(to=self.audio_duration)
            self.slider_start.set(0)
            self.slider_end.set(self.audio_duration)
            self.lbl_start.configure(text="Start: 0.0s")
            self.lbl_end.configure(text=f"Ende: {self.audio_duration:.1f}s")
            self.lbl_audio_status.configure(
                text=f"{os.path.basename(path)} ({self.audio_duration:.1f}s)",
                text_color="#38bdf8"
            )
            self.btn_audio_preview.configure(state="normal")

    def update_start_lbl(self, val):
        val = float(val)
        if val >= self.slider_end.get():
            val = max(0, self.slider_end.get() - 0.5)
            self.slider_start.set(val)
        self.lbl_start.configure(text=f"Start: {val:.1f}s")

    def update_end_lbl(self, val):
        val = float(val)
        if val <= self.slider_start.get():
            val = min(self.audio_duration, self.slider_start.get() + 0.5)
            self.slider_end.set(val)
        self.lbl_end.configure(text=f"Ende: {val:.1f}s")

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
        dur = max(0.1, end_t - start_t)

        temp_preview = os.path.join(tempfile.gettempdir(), "autoedit_preview_temp.wav")
        try:
            cmd = [
                self.ffmpeg_exe, "-y",
                "-ss", f"{start_t:.3f}",
                "-i", self.audio_path,
                "-t", f"{dur:.3f}",
                "-vn",
                "-c:a", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                temp_preview
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            pygame.mixer.music.load(temp_preview)
            pygame.mixer.music.play()
            self.is_preview_playing = True
            self.btn_audio_preview.configure(text="Pause", fg_color="#c026d3", hover_color="#9333ea")

            def check_finished():
                if self.is_preview_playing and not pygame.mixer.music.get_busy():
                    self.stop_audio_preview()
                elif self.is_preview_playing:
                    self.after(100, check_finished)

            self.after(100, check_finished)
        except Exception as e:
            self.show_error("Preview-Fehler", f"Konnte Audio-Vorschau nicht abspielen:\n{e}")
            self.stop_audio_preview()

    def stop_audio_preview(self):
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        self.is_preview_playing = False
        self.btn_audio_preview.configure(text="Vorhören", fg_color="#2b719e", hover_color="#1f5375")

    # =====================================================================
    # Cancellation & Post-Actions
    # =====================================================================
    def cancel_generation(self):
        self.is_cancelled = True
        self.set_status("Abbruch angefordert... Bereinige Prozesse.", 0.0)

    def open_output_folder(self):
        if self.last_rendered_path and os.path.exists(self.last_rendered_path):
            folder = os.path.dirname(self.last_rendered_path)
            subprocess.Popen(f'explorer /select,"{os.path.normpath(self.last_rendered_path)}"')
        else:
            self.show_error("Fehler", "Kein erstelltes Video gefunden.")

    def play_output_video(self):
        if self.last_rendered_path and os.path.exists(self.last_rendered_path):
            os.startfile(self.last_rendered_path)
        else:
            self.show_error("Fehler", "Video-Datei existiert nicht.")

    # =====================================================================
    # Generation Workflow
    # =====================================================================
    def start_generation(self):
        if not self.video_paths:
            self.show_error("Fehler", "Bitte wähle mindestens ein Video oder Bild aus.")
            return
        if not self.audio_path:
            self.show_error("Fehler", "Bitte wähle eine Musik-Datei aus.")
            return

        start_t = self.slider_start.get()
        end_t = self.slider_end.get()
        if end_t <= start_t:
            self.show_error("Fehler", "Das Ende der Musik muss nach dem Start liegen.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Video speichern als",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if not out_path:
            return

        self.is_cancelled = False
        self.btn_generate.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.post_action_frame.pack_forget()

        if self.shuffle_var.get():
            random.shuffle(self.video_paths)

        mode = self.mode_var.get()
        if mode == "manual":
            self.open_manual_recorder(start_t, end_t, out_path)
        else:
            self.set_status("Starte Beat-Analyse & Vorbereitung...", 0.05)
            threading.Thread(
                target=self.generate_edit,
                args=(start_t, end_t, out_path, None, mode),
                daemon=True
            ).start()

    # =====================================================================
    # Manual Recording Mode (Takt-Magnet)
    # =====================================================================
    def open_manual_recorder(self, start_t, end_t, out_path):
        total_duration = end_t - start_t
        temp_dir = tempfile.mkdtemp(prefix="autoedit_manual_")
        master_audio_wav = os.path.join(temp_dir, "master_audio.wav")

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
            text="Takt-Magnet aktiv (Klicks an Drum-Hits einrasten)",
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

        btn_start = ctk.CTkButton(rec_win, text="Aufnahme Starten", command=start_play, font=("Arial", 15, "bold"), height=40)
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
            else:
                crf_val = "16"
                preset_val = "veryfast"

            res_choice = self.res_var.get()

            # Effect controls & intensity
            fx_style = self.combo_fx_style.get()
            intensity = float(self.slider_fx_intensity.get()) / 100.0
            allow_shake = self.fx_shake.get()
            allow_bloom = self.fx_bloom_flash.get()
            allow_punch = self.fx_punch_zoom.get()
            allow_push = self.fx_slow_push.get()
            allow_rgb = self.fx_rgb_shock.get()
            allow_vignette = self.fx_vignette.get()
            allow_speedramp = self.fx_speedramp.get()

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
                    beat_frames = librosa.onset.onset_backtrack(beat_frames, energy=onset_env)
                    raw_beats = librosa.frames_to_time(beat_frames, sr=sr)

                    rms = librosa.feature.rms(y=y)[0]
                    mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.01

                    # Beat-Drop Detection: find maximum sub-bass energy jump after intro (>= 2.5s)
                    diff_low = np.diff(env_low)
                    min_f = int(2.5 * sr / 512)
                    if len(diff_low) > min_f + 10:
                        drop_frame = min_f + int(np.argmax(diff_low[min_f:]))
                        drop_raw_time = float(librosa.frames_to_time(drop_frame, sr=sr))
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

            # Precompute Vignette Mask
            vignette_mask = _build_vignette_mask(tw, th) if allow_vignette else None

            # Pre-build Spinning Token
            prepared_token = None
            if "Groß" in overlay_size_str or "Gross" in overlay_size_str:
                token_ratio = 0.45
            elif "Mittel" in overlay_size_str:
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
            # Modern High-Impact Edit Effect Engine
            # =========================================================
            def apply_pro_effects(clip, dur, is_impact=True, active_effects=None):
                if active_effects is None:
                    active_effects = {"shake", "bloom", "punch", "rgb", "push"}

                fd = min(0.14, dur)
                sd = min(0.20, dur)
                rd = min(0.09, dur)

                use_shake = allow_shake and ("shake" in active_effects) and is_impact
                use_bloom = allow_bloom and ("bloom" in active_effects) and is_impact
                use_punch = allow_punch and ("punch" in active_effects) and is_impact
                use_rgb = allow_rgb and ("rgb" in active_effects) and is_impact
                use_push = allow_push and ("push" in active_effects) and (not is_impact or dur > 0.6)

                def process_frame(gf, t):
                    frame = gf(t)

                    # 1. Spatial Transformation: Punch Zoom + Rotational Tilt + Recoil displacement
                    decay_zoom = ((max(0.0, 1.0 - (t / dur))) ** 3.0)
                    cur_zoom = 1.0 + (0.13 * intensity * decay_zoom) if use_punch else 1.0
                    if use_push:
                        cur_zoom = 1.0 + (0.04 * (t / dur))

                    ang = 0.0
                    dx, dy = 0.0, 0.0
                    if use_shake and t < sd:
                        decay_shake = ((max(0.0, 1.0 - (t / sd))) ** 2.5)
                        ang = np.sin(t * 38.0) * (2.2 * intensity) * decay_shake
                        dx = (np.sin(t * 52.0) * 16.0 + np.sin(t * 22.0) * 8.0) * intensity * decay_shake
                        dy = (np.cos(t * 44.0) * 12.0 + np.cos(t * 18.0) * 6.0) * intensity * decay_shake

                    if abs(ang) > 0.01 or cur_zoom > 1.002 or abs(dx) > 0.5 or abs(dy) > 0.5:
                        M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), ang, cur_zoom)
                        M[0, 2] += dx
                        M[1, 2] += dy
                        frame = cv2.warpAffine(frame, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                    # 2. Exposure Bloom Flash (Highlight-Burst mit Farbtemperatur)
                    if use_bloom and t < fd:
                        decay_fl = ((max(0.0, 1.0 - (t / fd))) ** 2.2) * intensity
                        gain = 1.0 + (0.90 * decay_fl)
                        warm_tint = np.array([255, 246, 225], dtype=np.float32)
                        boosted = frame.astype(np.float32) * gain + (warm_tint * (0.28 * decay_fl))
                        frame = np.clip(boosted, 0, 255).astype(np.uint8)

                    # 3. Radial Lens Chromatic Aberration (RGB-Linsenzerlegung)
                    if use_rgb and t < rd:
                        decay_rgb = ((max(0.0, 1.0 - (t / rd))) ** 2.0) * intensity
                        if decay_rgb > 0.02:
                            r_scale = 1.0 + (0.014 * decay_rgb)
                            b_scale = 1.0 - (0.014 * decay_rgb)
                            Mr = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), 0, r_scale)
                            Mb = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), 0, b_scale)
                            r_ch = cv2.warpAffine(frame[:, :, 0], Mr, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                            b_ch = cv2.warpAffine(frame[:, :, 2], Mb, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                            frame[:, :, 0] = r_ch
                            frame[:, :, 2] = b_ch

                    # 4. Cinematic Vignette
                    if allow_vignette and vignette_mask is not None:
                        frame = cv2.multiply(frame, vignette_mask, scale=1.0 / 255.0)

                    # 5. Cinematic S-Curve Contrast & Color Pop LUT
                    if apply_cinematic_lut:
                        frame = cv2.LUT(frame, CINEMATIC_LUT)

                    return frame

                return clip.fl(process_frame)

            # Spinning Overlay Compositor
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
                else:
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

                # Determine effect mode for this clip
                use_speedramp = allow_speedramp and not isinstance(source_clip, ImageClip) and (duration < 0.8) and (i % 3 == 0)
                if use_speedramp:
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

                # Determine active effects based on chosen style
                if "Impact-Combo" in fx_style:
                    # Coordinated impact combo stacked on beats
                    sub_clip = apply_pro_effects(sub_clip, duration, is_impact=True)
                elif "Zufällig" in fx_style:
                    all_possible = ["shake", "bloom", "punch", "rgb", "push"]
                    pick = {random.choice(all_possible)}
                    sub_clip = apply_pro_effects(sub_clip, duration, is_impact=True, active_effects=pick)
                else:
                    # Dezent & Sanft
                    sub_clip = apply_pro_effects(sub_clip, duration, is_impact=False, active_effects={"push", "bloom"})

                # Apply Spinning Overlay if trigger reached
                if has_overlay and cut_global_start >= active_overlay_start:
                    sub_clip = apply_spinning_overlay(sub_clip, i, cut_global_start, duration)

                # Audio handling
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
                    f.write(f"file '{os.path.abspath(tp).replace(os.sep, '/')}'\n")

            visual_concat_mp4 = os.path.join(temp_dir, "visual_concat.mp4")
            cmd_concat = [
                self.ffmpeg_exe, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_txt,
                "-c", "copy",
                visual_concat_mp4
            ]
            res_c = subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res_c.returncode != 0:
                cmd_concat_fallback = [
                    self.ffmpeg_exe, "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_txt,
                    "-c:v", "libx264",
                    "-preset", preset_val,
                    "-crf", crf_val,
                    visual_concat_mp4
                ]
                subprocess.run(cmd_concat_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            self.set_status("Schritt 3/3: Muxe unkomprimiertes Master-Audio & Fade-Out...", 0.95)

            audio_filter = f"afade=t=out:st={max(0, total_duration - 0.8):.2f}:d=0.8" if enable_fade_out else "anull"
            final_cmd = [
                self.ffmpeg_exe, "-y",
                "-i", visual_concat_mp4,
                "-i", master_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "320k",
                "-af", audio_filter,
                "-shortest",
                out_path
            ]
            subprocess.run(final_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            self.last_rendered_path = out_path
            self.set_status(f"Export erfolgreich abgeschlossen: {os.path.basename(out_path)}", 1.0)
            self.show_info("Fertig", f"Video erfolgreich gespeichert unter:\n{out_path}")
            self.after(0, lambda: self.post_action_frame.pack(fill="x", padx=20, pady=(6, 10)))

        except InterruptedError:
            self.set_status("Rendern abgebrochen. Keine Datei erstellt.", 0.0)
        except Exception as e:
            traceback.print_exc()
            self.set_status(f"Fehler: {e}", 0.0)
            self.show_error("Fehler", f"Fehler beim Rendern:\n{traceback.format_exc()}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.after(0, lambda: self.btn_generate.configure(state="normal"))
            self.after(0, lambda: self.btn_cancel.configure(state="disabled"))


if __name__ == "__main__":
    app = EditMakerApp()
    app.mainloop()
