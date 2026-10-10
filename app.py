import customtkinter as ctk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, ColorClip, vfx, ImageClip
import os
import threading
import random
import traceback
import time
import pygame

# Patch for Pillow 10+ compatibility with MoviePy 1.0.3
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    try:
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Patch for MoviePy 1.0.3 "failed to read the first frame" bug (WhatsApp / variable framerate / delayed PTS videos)
import moviepy.video.io.ffmpeg_reader as ffr
_orig_read_frame = ffr.FFMPEG_VideoReader.read_frame
def _robust_read_frame(self):
    try:
        return _orig_read_frame(self)
    except IOError as e:
        if "failed to read the first frame" in str(e):
            # Try seeking slightly into the video (0.1s) in case PTS / keyframe starts later
            try:
                self.initialize(0.1)
                self.pos = int(self.fps * 0.1 + 0.00001) + 1
                return _orig_read_frame(self)
            except Exception:
                pass
            # If still failing, produce a blank frame so the pipeline doesn't crash
            w, h = self.size
            depth = getattr(self, 'depth', 3)
            blank = np.zeros((h, w, depth), dtype='uint8')
            self.lastread = blank
            return blank
        raise
ffr.FFMPEG_VideoReader.read_frame = _robust_read_frame

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class EditMakerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Auto Beat-Sync Edit Maker")
        self.geometry("650x680")
        self.resizable(False, False)

        self.video_paths = []
        self.audio_path = None
        self.audio_duration = 0

        # Title
        self.lbl_title = ctk.CTkLabel(self, text="Auto Edit Maker", font=("Arial", 28, "bold"))
        self.lbl_title.pack(pady=(20, 10))

        # Video Selection
        self.btn_select_videos = ctk.CTkButton(self, text="Select Video Clips", command=self.select_videos)
        self.btn_select_videos.pack(pady=5)

        self.lbl_videos = ctk.CTkLabel(self, text="0 videos selected", text_color="gray")
        self.lbl_videos.pack(pady=(0, 5))

        # Shuffle option
        self.shuffle_var = ctk.BooleanVar(value=False)
        self.chk_shuffle = ctk.CTkCheckBox(self, text="Shuffle File Order", variable=self.shuffle_var)
        self.chk_shuffle.pack(pady=(0, 10))

        # Audio Selection
        self.btn_select_audio = ctk.CTkButton(self, text="Select Audio (Song)", command=self.select_audio)
        self.btn_select_audio.pack(pady=5)

        self.lbl_audio = ctk.CTkLabel(self, text="No audio selected", text_color="gray")
        self.lbl_audio.pack(pady=(0, 15))

        # Audio Trim Sliders
        self.frame_sliders = ctk.CTkFrame(self)
        self.frame_sliders.pack(pady=10, fill="x", padx=40)
        
        self.lbl_sliders_title = ctk.CTkLabel(self.frame_sliders, text="Audio Selection (Start & End Time)", font=("Arial", 14, "bold"))
        self.lbl_sliders_title.pack(pady=5)

        self.lbl_start = ctk.CTkLabel(self.frame_sliders, text="Start Time: 0.0s")
        self.lbl_start.pack()
        self.slider_start = ctk.CTkSlider(self.frame_sliders, from_=0, to=100, command=self.update_start_lbl)
        self.slider_start.set(0)
        self.slider_start.pack(fill="x", padx=20, pady=(0, 10))

        self.lbl_end = ctk.CTkLabel(self.frame_sliders, text="End Time: 0.0s")
        self.lbl_end.pack()
        self.slider_end = ctk.CTkSlider(self.frame_sliders, from_=0, to=100, command=self.update_end_lbl)
        self.slider_end.pack(fill="x", padx=20, pady=(0, 10))

        # Mode Switch
        self.mode_var = ctk.StringVar(value="auto")
        self.radio_auto = ctk.CTkRadioButton(self, text="Auto Mode (Kick Detection - Takt)", variable=self.mode_var, value="auto")
        self.radio_auto.pack(pady=(5, 3))
        self.radio_onsets = ctk.CTkRadioButton(self, text="Freeform Beat-Sync (Alle Beats/Hits - kein fester Takt)", variable=self.mode_var, value="onsets")
        self.radio_onsets.pack(pady=(3, 3))
        self.radio_manual = ctk.CTkRadioButton(self, text="Manual Mode (Press Spacebar to Cut)", variable=self.mode_var, value="manual")
        self.radio_manual.pack(pady=(3, 10))

        # Orientation Selection
        self.frame_orient = ctk.CTkFrame(self)
        self.frame_orient.pack(pady=5)
        self.lbl_orient = ctk.CTkLabel(self.frame_orient, text="Output Format:", font=("Arial", 13, "bold"))
        self.lbl_orient.pack(side="left", padx=(10, 8))
        self.orient_var = ctk.StringVar(value="original")
        for label, val in [("Original", "original"), ("Hochkant (9:16)", "portrait"), ("Quer (16:9)", "landscape")]:
            ctk.CTkRadioButton(self.frame_orient, text=label, variable=self.orient_var, value=val).pack(side="left", padx=6)

        # Generate Button
        self.btn_generate = ctk.CTkButton(self, text="Generate Edit", command=self.start_generation, 
                                          fg_color="green", hover_color="darkgreen", height=40, font=("Arial", 16, "bold"))
        self.btn_generate.pack(pady=10)

        # Status Label
        self.lbl_status = ctk.CTkLabel(self, text="Ready", text_color="yellow")
        self.lbl_status.pack()

    def update_start_lbl(self, val):
        self.lbl_start.configure(text=f"Start Time: {val:.1f}s")
        if val > self.slider_end.get():
            self.slider_end.set(val)
            self.lbl_end.configure(text=f"End Time: {val:.1f}s")

    def update_end_lbl(self, val):
        self.lbl_end.configure(text=f"End Time: {val:.1f}s")
        if val < self.slider_start.get():
            self.slider_start.set(val)
            self.lbl_start.configure(text=f"Start Time: {val:.1f}s")

    def select_videos(self):
        paths = filedialog.askopenfilenames(title="Select Media (Videos/Images)", filetypes=[("Media Files", "*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg *.webp"), ("All Files", "*.*")])
        if paths:
            self.video_paths = list(paths)
            self.lbl_videos.configure(text=f"{len(self.video_paths)} media files selected", text_color="white")

    def select_audio(self):
        path = filedialog.askopenfilename(title="Select Audio", filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.m4a *.aac"), ("All Files", "*.*")])
        if path:
            self.audio_path = path
            filename = os.path.basename(path)
            self.lbl_audio.configure(text=f"Audio: {filename}", text_color="white")
            
            try:
                clip = AudioFileClip(self.audio_path)
                self.audio_duration = clip.duration
                clip.close()
                
                self.slider_start.configure(to=self.audio_duration)
                self.slider_end.configure(to=self.audio_duration)
                self.slider_start.set(0)
                self.slider_end.set(self.audio_duration)
                self.update_start_lbl(0)
                self.update_end_lbl(self.audio_duration)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load audio: {e}")

    def start_generation(self):
        if not self.video_paths:
            messagebox.showwarning("Warning", "Please select at least one video clip.")
            return
        if not self.audio_path:
            messagebox.showwarning("Warning", "Please select an audio file.")
            return
            
        start_t = self.slider_start.get()
        end_t = self.slider_end.get()
        
        if start_t >= end_t:
            messagebox.showwarning("Warning", "End time must be greater than start time.")
            return

        # --- Pre-validate all selected video/image files ---
        self.lbl_status.configure(text="Validating files...")
        self.update()
        valid_paths = []
        bad_paths = []
        for vp in self.video_paths:
            ext = os.path.splitext(vp)[1].lower()
            try:
                if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                    PIL.Image.open(vp).verify()
                else:
                    test = VideoFileClip(vp)
                    # Try reading first frame to catch "failed to read first frame" errors
                    _ = test.get_frame(0)
                    test.close()
                valid_paths.append(vp)
            except Exception as e:
                bad_paths.append(os.path.basename(vp))

        if bad_paths:
            messagebox.showwarning(
                "Skipping bad files",
                "The following files could not be read and will be skipped:\n\n" +
                "\n".join(bad_paths)
            )
        if not valid_paths:
            messagebox.showerror("Error", "No valid media files found. Aborting.")
            self.lbl_status.configure(text="Ready")
            return
        self.video_paths = valid_paths
        if self.shuffle_var.get():
            random.shuffle(self.video_paths)
        self.lbl_videos.configure(text=f"{len(self.video_paths)} media files selected", text_color="white")

        out_path = filedialog.asksaveasfilename(title="Save Video", defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if not out_path:
            self.lbl_status.configure(text="Ready")
            return

        self.btn_generate.configure(state="disabled")
        
        mode = self.mode_var.get()
        if mode in ("auto", "onsets"):
            self.lbl_status.configure(text="Processing... This will take a while.")
            self.update()
            threading.Thread(target=self.generate_edit, args=(start_t, end_t, out_path, None, mode), daemon=True).start()
        else:
            self.start_manual_recording(start_t, end_t, out_path)

    def start_manual_recording(self, start_t, end_t, out_path):
        self.lbl_status.configure(text="Preparing audio for manual mode...")
        self.update()
        
        pygame.mixer.init()
        temp_wav = "temp_manual.wav"
        
        try:
            audio_clip = AudioFileClip(self.audio_path).subclip(start_t, end_t)
            audio_clip.write_audiofile(temp_wav, logger=None)
            audio_clip.close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare audio: {e}")
            self.btn_generate.configure(state="normal")
            return
            
        self.manual_times = []
        
        rec_win = ctk.CTkToplevel(self)
        rec_win.title("Manual Cut Recording")
        rec_win.geometry("450x250")
        rec_win.transient(self)
        rec_win.grab_set()
        
        lbl_info = ctk.CTkLabel(rec_win, text="Press SPACE, any arrow key, or LMB/RMB to record a cut.\nClose this window when done to start generating.", font=("Arial", 14))
        lbl_info.pack(pady=20)
        
        lbl_cuts = ctk.CTkLabel(rec_win, text="Cuts recorded: 0", font=("Arial", 18, "bold"), text_color="yellow")
        lbl_cuts.pack(pady=10)
        
        try:
            pygame.mixer.music.load(temp_wav)
        except Exception as e:
            messagebox.showerror("Error", f"Pygame couldn't load audio: {e}")
            rec_win.destroy()
            self.btn_generate.configure(state="normal")
            return
            
        is_playing = False
        
        def record_cut(event=None):
            if not is_playing:
                return
            current_time = pygame.mixer.music.get_pos() / 1000.0
            if current_time > 0:
                self.manual_times.append(current_time)
                lbl_cuts.configure(text=f"Cuts recorded: {len(self.manual_times)}")

        # Spacebar + all arrow keys + left/right mouse button
        for key in ("<space>", "<Left>", "<Right>", "<Up>", "<Down>"):
            rec_win.bind(key, record_cut)
        rec_win.bind("<Button-1>", record_cut)   # LMB
        rec_win.bind("<Button-3>", record_cut)   # RMB
        
        def start_play():
            nonlocal is_playing
            is_playing = True
            btn_start.configure(state="disabled")
            pygame.mixer.music.play()
            rec_win.focus_set() # Focus back to capture spacebar
            
        btn_start = ctk.CTkButton(rec_win, text="Start Playback", command=start_play, font=("Arial", 16, "bold"), height=40)
        btn_start.pack(pady=10)
        
        def on_close():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            rec_win.destroy()
            
            try:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except:
                pass
                
            if len(self.manual_times) > 0:
                beat_times = sorted(self.manual_times)
                if beat_times[0] > 0.05: # Prevent extremely small first cuts
                    beat_times.insert(0, 0.0)
                else:
                    beat_times[0] = 0.0
                    
                if beat_times[-1] < (end_t - start_t):
                    beat_times.append(end_t - start_t)
                    
                self.lbl_status.configure(text="Generating edit from manual cuts...")
                threading.Thread(target=self.generate_edit, args=(start_t, end_t, out_path, beat_times), daemon=True).start()
            else:
                self.lbl_status.configure(text="Recording cancelled (no cuts).")
                self.btn_generate.configure(state="normal")
                
        rec_win.protocol("WM_DELETE_WINDOW", on_close)

    def generate_edit(self, start_t, end_t, out_path, custom_beat_times=None, mode="auto"):
        try:
            orientation = self.orient_var.get()  # "original", "portrait", "landscape"

            if custom_beat_times is None:
                total_duration = end_t - start_t
                # Use librosa to load audio
                y, sr = librosa.load(self.audio_path, sr=None, offset=start_t, duration=total_duration)
                
                # Isolate percussive parts to ignore vocals/melodies
                y_percussive = librosa.effects.percussive(y, margin=3.0)

                if mode == "onsets":
                    self.lbl_status.configure(text="Step 1/3: Detecting freeform beats & hits (kein fester Takt)...")
                    # Full percussive onset strength (low + mid hits)
                    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
                    
                    # Detect all genuine onset peaks directly (natural hits/beats, independent of regular tempo)
                    # wait=7 corresponds to ~0.16s minimum spacing between cuts (prevents ultra-flickering)
                    hop_len = 512
                    onset_frames = librosa.onset.onset_detect(
                        onset_envelope=onset_env, sr=sr, hop_length=hop_len,
                        backtrack=True, wait=7, pre_max=5, post_max=5, pre_avg=7, post_avg=7, delta=0.07
                    )
                    beat_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_len)

                    # Free arrays early
                    del y, y_percussive, onset_env, onset_frames

                    # Ensure cuts cover from 0.0 to total duration
                    beat_times = np.insert(beat_times, 0, 0.0)
                    if len(beat_times) == 1 or beat_times[-1] < total_duration:
                        beat_times = np.append(beat_times, total_duration)

                    durations = np.diff(beat_times)
                else:
                    self.lbl_status.configure(text="Step 1/3: Analyzing audio beats and song intro (Takt-Modus)...")
                    # Calculate onset strength focusing heavily on low frequencies (<250Hz for kicks)
                    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, fmax=250)
                    
                    # Track beats based on the kick onsets (regular meter/tempo)
                    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                    
                    # Calculate RMS energy to detect the intro/outro (low energy parts)
                    rms = librosa.feature.rms(y=y)[0]
                    mean_rms = np.mean(rms)
                    
                    # Free audio arrays early
                    del y, y_percussive, onset_env, beat_frames

                    # Ensure we cover the full duration with cuts
                    beat_times = np.insert(beat_times, 0, 0.0)
                    if len(beat_times) == 1 or beat_times[-1] < total_duration:
                        beat_times = np.append(beat_times, total_duration)
                        
                    # Merge beats during low energy (Intro/Outro recognition for longer clips)
                    merged_beats = [beat_times[0]]
                    current_beat_idx = 0
                    while current_beat_idx < len(beat_times) - 1:
                        next_beat_idx = current_beat_idx + 1
                        
                        # Check energy in this interval
                        start_f = librosa.time_to_frames(beat_times[current_beat_idx], sr=sr)
                        end_f = librosa.time_to_frames(beat_times[next_beat_idx], sr=sr)
                        
                        if end_f > start_f:
                            segment_rms = np.mean(rms[start_f:end_f])
                        else:
                            segment_rms = mean_rms
                            
                        # If energy is low (< 70% of mean), merge up to 4 beats to make longer clips
                        if segment_rms < mean_rms * 0.7:
                            next_beat_idx = min(len(beat_times) - 1, current_beat_idx + 4)
                            
                        merged_beats.append(beat_times[next_beat_idx])
                        current_beat_idx = next_beat_idx

                    del rms, beat_times
                    durations = np.diff(merged_beats)
            else:
                self.lbl_status.configure(text="Step 1/3: Using manual cuts...")
                durations = np.diff(custom_beat_times)

            self.lbl_status.configure(text="Step 2/3: Preparing video clips & effects...")

            # --- Determine output canvas size from FIRST clip only ---
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

            if orientation == "portrait":
                tw, th = 1080, 1920
            elif orientation == "landscape":
                tw, th = 1920, 1080
            else:
                tw, th = orig_w, orig_h

            target_size = (tw, th)

            # Per-path state: video resume positions & portrait crop cycling
            video_positions = {}  # path -> next start time (s)
            image_crop_idx  = {}  # path -> next crop-zone index (cycles top/center/bottom)

            # Portrait crop zones as relative y positions (top / center / bottom thirds)
            PORTRAIT_CROP_ZONES = [0.2, 0.5, 0.8]

            # --- Helper: crop/resize a clip to fill target canvas ---
            def fit_clip(clip, target_w, target_h, path=None):
                cw, ch = clip.size
                target_ratio = target_w / target_h
                clip_ratio   = cw / ch
                if clip_ratio > target_ratio:
                    resized = clip.resize(height=target_h)
                else:
                    resized = clip.resize(width=target_w)

                x_center = resized.w / 2
                y_center = resized.h / 2

                # Portrait clip in landscape canvas → cycle crop zones vertically
                if path is not None and clip_ratio < 1.0 and target_ratio > 1.0:
                    idx      = image_crop_idx.get(path, 0)
                    y_center = resized.h * PORTRAIT_CROP_ZONES[idx % len(PORTRAIT_CROP_ZONES)]
                    image_crop_idx[path] = idx + 1

                return resized.crop(
                    x_center=x_center,
                    y_center=y_center,
                    width=target_w,
                    height=target_h,
                )

            # --- Helper: RGB glitch effect ---
            def apply_glitch(image):
                glitched = np.copy(image)
                if random.random() < 0.2:
                    shift = random.randint(-15, 15)
                    glitched[:, :, 0] = np.roll(glitched[:, :, 0], shift, axis=1)
                return glitched

            # -------------------------------------------------------
            # Build clip list by opening each source ONE AT A TIME,
            # processing the segment, writing to a temp file, then
            # closing it immediately to free RAM.
            # -------------------------------------------------------
            temp_files = []
            clip_idx = 0
            video_paths = self.video_paths  # local reference

            # Helper: open a source clip from path (pre-rescaling large images saves huge memory and CPU)
            def open_source(path):
                ext_ = os.path.splitext(path)[1].lower()
                if ext_ in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                    img_ = PIL.Image.open(path).convert("RGB")
                    # If image is massively larger than target canvas, pre-scale with thumbnail
                    if img_.width > tw * 1.5 or img_.height > th * 1.5:
                        scale = max(tw / img_.width, th / img_.height)
                        nw, nh = max(1, int(img_.width * scale)), max(1, int(img_.height * scale))
                        img_ = img_.resize((nw, nh), resample=PIL.Image.BILINEAR)
                    c_ = ImageClip(np.array(img_))
                    img_.close()
                else:
                    c_ = VideoFileClip(path)
                return c_

            # Helper: cut a source clip to a given duration, resuming for videos
            def cut_source(src, dur, path=None):
                if isinstance(src, ImageClip):
                    return src.set_duration(dur)
                # Determine start position
                if path is not None and path in video_positions:
                    cs = video_positions[path]
                    if cs >= src.duration - 0.05:
                        cs = 0.0  # wrap around to start
                else:
                    # First use: pick random start so different runs vary
                    max_s = max(0, src.duration - dur)
                    cs = random.uniform(0, max_s) if max_s > 0 else 0.0
                end_s = min(cs + dur, src.duration)
                sc = src.subclip(cs, end_s)
                # Store position for next use of this path
                if path is not None:
                    next_pos = end_s if end_s < src.duration - 0.05 else 0.0
                    video_positions[path] = next_pos
                if sc.duration < dur:
                    sc = sc.fx(vfx.loop, duration=dur)
                return sc


            i = 0
            while i < len(durations):
                duration = float(durations[i])
                if duration < 0.1:
                    duration = 0.1

                source_clip = open_source(video_paths[clip_idx])

                # Cut + fit the primary segment
                cur_path = video_paths[clip_idx]
                sub_clip = fit_clip(cut_source(source_clip, duration, cur_path), tw, th, cur_path)

                # --- Effect selection ---
                available_after = len(durations) - i - 1
                do_collage_16 = (available_after >= 4) and (random.random() < 0.025)
                do_collage    = (not do_collage_16) and (available_after >= 1) and (random.random() < 0.05)
                do_spin       = (not do_collage_16) and (not do_collage) and (random.random() < 0.05)

                if do_collage_16:
                    # ── 16-GRID COLLAGE EFFECT (4x4 Grid) ───────────────────────
                    # Background = current clip (fullscreen)
                    # Over 4 beats, 4 tiles appear per beat to fill a 4x4 grid (16 tiles total).
                    num_steps = 4
                    step_durs = [max(float(durations[i + j]), 0.1) for j in range(1, num_steps + 1)]
                    total_dur = duration + sum(step_durs)

                    bg = fit_clip(source_clip, tw, th, cur_path).set_duration(total_dur)
                    gw, gh = tw // 4, th // 4
                    layers = [bg]
                    extra_sources = []

                    # Precompute 4x4 grid positions [(x, y), ...]
                    grid_positions = [
                        (col * gw, row * gh)
                        for row in range(4)
                        for col in range(4)
                    ]

                    t_offset = duration
                    tile_idx = 0
                    for step in range(num_steps):
                        # Remaining duration from this step until the end of the collage
                        step_rem_dur = sum(step_durs[step:])
                        for _ in range(4):
                            t_vp = video_paths[(clip_idx + 1 + tile_idx) % len(video_paths)]
                            t_src = open_source(t_vp)
                            extra_sources.append(t_src)

                            tile_clip = fit_clip(cut_source(t_src, step_rem_dur, t_vp), gw, gh, t_vp)
                            tile_clip = tile_clip.without_audio().set_start(t_offset).set_position(grid_positions[tile_idx])
                            layers.append(tile_clip)
                            tile_idx += 1

                        t_offset += step_durs[step]

                    sub_clip = CompositeVideoClip(layers, size=(tw, th)).without_audio()

                    temp_path = f"_temp_seg_{len(temp_files):04d}.mp4"
                    sub_clip.write_videofile(temp_path, fps=30, codec="libx264", threads=4, logger=None,
                                             ffmpeg_params=["-preset", "ultrafast", "-crf", "18", "-tune", "fastdecode"])
                    temp_files.append(temp_path)

                    source_clip.close()
                    for es in extra_sources:
                        es.close()
                    try:
                        sub_clip.close()
                    except Exception:
                        pass

                    self.lbl_status.configure(
                        text=f"Step 2/3: Processing clip {len(temp_files)}/{len(durations)} (16-collage)..."
                    )

                    clip_idx = (clip_idx + 1 + 16) % len(video_paths)
                    i += 1 + num_steps

                elif do_collage:
                    # ── COLLAGE EFFECT ──────────────────────────────────────────
                    # Background = current clip (fullscreen, full collage duration)
                    # Then up to 4 quadrant clips pop in one-by-one at each beat.
                    num_quads = min(4, available_after)
                    quad_durs = [max(float(durations[i + j]), 0.1) for j in range(1, num_quads + 1)]
                    total_dur = duration + sum(quad_durs)

                    bg = fit_clip(source_clip, tw, th, cur_path).set_duration(total_dur)
                    qw, qh = tw // 2, th // 2
                    quad_positions = [(0, 0), (tw // 2, 0), (0, th // 2), (tw // 2, th // 2)]
                    layers = [bg]
                    extra_sources = []

                    t_offset = duration
                    for q in range(num_quads):
                        q_vp = video_paths[(clip_idx + 1 + q) % len(video_paths)]
                        q_src = open_source(q_vp)
                        extra_sources.append(q_src)

                        # This quadrant stays visible for the remaining duration
                        q_dur = sum(quad_durs[q:])
                        q_clip = fit_clip(cut_source(q_src, q_dur, q_vp), qw, qh, q_vp)
                        q_clip = q_clip.without_audio().set_start(t_offset).set_position(quad_positions[q])
                        layers.append(q_clip)
                        t_offset += quad_durs[q]

                    sub_clip = CompositeVideoClip(layers, size=(tw, th)).without_audio()

                    temp_path = f"_temp_seg_{len(temp_files):04d}.mp4"
                    sub_clip.write_videofile(temp_path, fps=30, codec="libx264", threads=4, logger=None,
                                             ffmpeg_params=["-preset", "ultrafast", "-crf", "18", "-tune", "fastdecode"])
                    temp_files.append(temp_path)

                    source_clip.close()
                    for es in extra_sources:
                        es.close()
                    try:
                        sub_clip.close()
                    except Exception:
                        pass

                    self.lbl_status.configure(
                        text=f"Step 2/3: Processing clip {len(temp_files)}/{len(durations)} (collage)..."
                    )

                    # Advance past background + all quadrant beats we consumed
                    clip_idx = (clip_idx + 1 + num_quads) % len(video_paths)
                    i += 1 + num_quads

                elif do_spin:
                    # ── SPIN EFFECT ─────────────────────────────────────────────
                    # Clip rotates a full 360° over its duration.
                    # We expand the canvas temporarily so corners don't clip, then
                    # crop back to the target size at the end.
                    def make_spin_frame(get_frame, t, dur, diag):
                        angle = (t / dur) * 360.0
                        img = PIL.Image.fromarray(get_frame(t))
                        rotated = img.rotate(-angle, resample=PIL.Image.BILINEAR, expand=False)
                        return np.array(rotated)

                    spin_dur = duration
                    spin_clip = sub_clip.fl(lambda gf, t: make_spin_frame(gf, t, spin_dur, 0))
                    # Scale up slightly so the black corners from rotation are hidden
                    spin_clip = spin_clip.resize(1.42)
                    spin_clip = spin_clip.crop(
                        x_center=spin_clip.w / 2, y_center=spin_clip.h / 2,
                        width=tw, height=th,
                    ).without_audio()

                    temp_path = f"_temp_seg_{len(temp_files):04d}.mp4"
                    spin_clip.write_videofile(temp_path, fps=30, codec="libx264", threads=4, logger=None,
                                              ffmpeg_params=["-preset", "ultrafast", "-crf", "18", "-tune", "fastdecode"])
                    temp_files.append(temp_path)

                    source_clip.close()
                    try:
                        spin_clip.close()
                    except Exception:
                        pass

                    self.lbl_status.configure(
                        text=f"Step 2/3: Processing clip {len(temp_files)}/{len(durations)} (spin)..."
                    )

                    clip_idx = (clip_idx + 1) % len(video_paths)
                    i += 1

                else:
                    # ── NORMAL EFFECTS ───────────────────────────────────────────
                    if duration >= 1.0 and random.random() < 0.4:
                        seg_dur = duration / 4.0
                        parts = []
                        for part_idx in range(4):
                            zoom_factor = 1.0 + (part_idx * 0.06)
                            part_start = part_idx * seg_dur
                            part_end = min((part_idx + 1) * seg_dur, duration)
                            part_clip = sub_clip.subclip(part_start, part_end)
                            if zoom_factor > 1.0:
                                zoomed = part_clip.resize(zoom_factor)
                                part_clip = zoomed.crop(
                                    x_center=zoomed.w / 2, y_center=zoomed.h / 2,
                                    width=tw, height=th,
                                )
                            parts.append(part_clip)
                        sub_clip = concatenate_videoclips(parts, method="chain")
                    else:
                        effect = random.randint(0, 7)
                        if effect == 1:
                            flash = ColorClip(size=target_size, color=[255, 255, 255]).set_duration(min(0.1, duration)).set_opacity(0.4)
                            sub_clip = CompositeVideoClip([sub_clip, flash])
                        elif effect == 2:
                            zoomed = sub_clip.resize(1.1)
                            sub_clip = zoomed.crop(x_center=zoomed.w / 2, y_center=zoomed.h / 2, width=tw, height=th)
                        elif effect == 3:
                            sub_clip = sub_clip.fx(vfx.mirror_x)
                        elif effect == 4:
                            sub_clip = sub_clip.fx(vfx.speedx, 1.2)
                        elif effect == 5:
                            sub_clip = sub_clip.fl_image(apply_glitch)
                        elif effect == 6:
                            sub_clip = sub_clip.fx(vfx.fadein, min(0.3, duration))
                        elif effect == 7:
                            # 2x quick zoom in and zoom out (beat bounce / pulse effect)
                            def double_zoom(get_frame, t):
                                phase = (t / duration) * 2.0 * np.pi * 2.0  # 2 full waves
                                s = 1.0 + 0.18 * max(0.0, np.sin(phase))
                                frame = get_frame(t)
                                img = PIL.Image.fromarray(frame)
                                nw, nh = int(round(tw * s)), int(round(th * s))
                                img_resized = img.resize((nw, nh), resample=PIL.Image.BILINEAR)
                                left = (nw - tw) // 2
                                top = (nh - th) // 2
                                cropped = img_resized.crop((left, top, left + tw, top + th))
                                return np.array(cropped)

                            sub_clip = sub_clip.fl(double_zoom)

                    sub_clip = sub_clip.without_audio()

                    temp_path = f"_temp_seg_{len(temp_files):04d}.mp4"
                    sub_clip.write_videofile(temp_path, fps=30, codec="libx264", threads=4, logger=None,
                                             ffmpeg_params=["-preset", "ultrafast", "-crf", "18", "-tune", "fastdecode"])
                    temp_files.append(temp_path)

                    source_clip.close()
                    try:
                        sub_clip.close()
                    except Exception:
                        pass

                    self.lbl_status.configure(
                        text=f"Step 2/3: Processing clip {len(temp_files)}/{len(durations)}..."
                    )

                    clip_idx = (clip_idx + 1) % len(video_paths)
                    i += 1

            self.lbl_status.configure(text="Step 3/3: Rendering final video... (See terminal)")

            # Load temp segments (just references, no full decode into RAM)
            final_clips = [VideoFileClip(p) for p in temp_files]
            final_video = concatenate_videoclips(final_clips, method="chain")

            audio_clip = AudioFileClip(self.audio_path).subclip(start_t, end_t)
            final_video = final_video.set_audio(audio_clip)

            final_video.write_videofile(
                out_path, fps=30,
                codec="libx264",
                audio_codec="aac",
                threads=6,
                ffmpeg_params=["-preset", "fast", "-crf", "18"],
            )

            # Cleanup
            audio_clip.close()
            final_video.close()
            for c in final_clips:
                c.close()
            for p in temp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass

            self.lbl_status.configure(text="Finished successfully!")
            messagebox.showinfo("Success", f"Edit generated successfully and saved to:\n{out_path}")

        except Exception as e:
            err = traceback.format_exc()
            print(err)
            self.lbl_status.configure(text="Error occurred! Check terminal for details.")
            messagebox.showerror("Error", f"An error occurred:\n{e}")
        finally:
            self.btn_generate.configure(state="normal")
            # Emergency cleanup of temp files if error occurred
            for p in locals().get("temp_files", []):
                try:
                    os.remove(p)
                except Exception:
                    pass

if __name__ == "__main__":
    app = EditMakerApp()
    app.mainloop()
