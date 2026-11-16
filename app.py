import streamlit as st
import os
import json
import subprocess
import textwrap
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

st.set_page_config(page_title="AI Viral Clip Studio", page_icon="🎬", layout="centered")

# Custom Colorful CSS Styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF007F, #FF8C00, #00F2FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #FF007F 0%, #7928CA 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        width: 100%;
        box-shadow: 0 4px 15px rgba(255, 0, 127, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(121, 40, 202, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🎬 AI Viral Clip Studio (9:16)</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Turn long videos into viral Shorts & Reels with AI auto-captions</div>', unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.markdown("## ⚙️ Studio Settings")
api_key = st.sidebar.text_input("🔑 Enter Gemini API Key:", type="password", placeholder="AQ.Ab8RN6...")
caption_style = st.sidebar.selectbox("🎨 Caption Color Theme:", ["Vibrant Multi-Color", "Classic Yellow", "Neon Cyan"])
st.sidebar.markdown("---")
st.sidebar.info("✨ **Pro Tip:** Upload videos under 10 minutes for ultra-fast rendering.")

uploaded_file = st.file_uploader("📤 Upload Your Video (MP4 / MOV)", type=["mp4", "mov"])
video_path = "input_video.mp4"
audio_path = "temp_audio.mp3"

def get_smart_crop_x(video_path, start_t, end_t):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_width = int(height * (9 / 16))
    center_x = width // 2
    
    try:
        mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((start_t + (end_t - start_t)/2) * fps))
        ret, frame = cap.read()
        if ret:
            results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                face_center = int((bbox.xmin + bbox.width / 2) * width)
                center_x = max(target_width // 2, min(width - target_width // 2, face_center))
    except Exception:
        pass
    finally:
        cap.release()
            
    return max(0, center_x - (target_width // 2)), target_width

def create_caption_image(text, width, color_hex):
    """Creates a transparent PNG caption using Pillow (Bypasses ImageMagick completely)"""
    img_h = 260
    img = Image.new("RGBA", (width, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    font_size = max(28, int(width * 0.068))
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, font_size)
            break
    if font is None:
        font = ImageFont.load_default()

    wrapped_lines = textwrap.wrap(text, width=18)
    y_offset = 20
    for line in wrapped_lines[:3]:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (width - text_w) // 2
        
        # Draw thick black outline
        stroke_w = 4
        for dx in range(-stroke_w, stroke_w + 1):
            for dy in range(-stroke_w, stroke_w + 1):
                draw.text((x + dx, y_offset + dy), line, font=font, fill="black")
        
        # Draw main colored text
        draw.text((x, y_offset), line, font=font, fill=color_hex)
        y_offset += text_h + 14

    return np.array(img)

if st.button("✨ Generate Viral 9:16 Clip"):
    if not api_key:
        st.error("🚨 Please enter your Gemini API Key in the left sidebar!")
        st.stop()
    if not uploaded_file:
        st.warning("⚠️ Please upload a video file first.")
        st.stop()

    with st.spinner("🎵 Step 1/3: Extracting audio track..."):
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", audio_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    with st.spinner("🧠 Step 2/3: AI is hunting for the most viral moment & generating English captions..."):
        try:
            client = genai.Client(api_key=api_key.strip())
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            prompt = """
            Listen to this audio and find the single most engaging, viral-worthy segment (between 15 and 40 seconds long).
            Generate punchy, short English subtitles/captions (2 to 4 words per chunk) with accurate timestamps for that segment.
            """
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "start": {"type": "NUMBER"},
                            "end": {"type": "NUMBER"},
                            "title": {"type": "STRING"},
                            "viral_score": {"type": "INTEGER"},
                            "captions": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "start": {"type": "NUMBER"},
                                        "end": {"type": "NUMBER"},
                                        "text": {"type": "STRING"}
                                    },
                                    "required": ["start", "end", "text"]
                                }
                            }
                        },
                        "required": ["start", "end", "title", "viral_score", "captions"]
                    }
                )
            )
            clip_data = json.loads(response.text)
            if isinstance(clip_data, list):
                clip_data = clip_data[0]
        except Exception as e:
            st.error(f"❌ Gemini API Error: {str(e)}")
            st.stop()

    with st.spinner("🎬 Step 3/3: Smart-cropping to 9:16 & rendering colorful captions..."):
        try:
            full_video = VideoFileClip(video_path)
            max_dur = full_video.duration
            
            start_t = max(0.0, float(clip_data.get("start", 0)))
            end_t = min(max_dur, float(clip_data.get("end", min(30.0, max_dur))))
            if end_t <= start_t:
                start_t = 0.0
                end_t = min(30.0, max_dur)
            
            video = full_video.subclip(start_t, end_t)
            x1, target_w = get_smart_crop_x(video_path, start_t, end_t)
            cropped_video = video.crop(x1=x1, y1=0, width=target_w, height=video.h)
            
            if caption_style == "Vibrant Multi-Color":
                colors = ['#FFD700', '#00FFFF', '#FFFFFF', '#FF69B4']
            elif caption_style == "Neon Cyan":
                colors = ['#00FFFF', '#FFFFFF']
            else:
                colors = ['#FFD700']

            caption_clips = [cropped_video]
            for idx, cap_item in enumerate(clip_data.get("captions", [])):
                c_start = max(0.0, float(cap_item.get("start", 0)) - start_t)
                c_end = min(end_t - start_t, float(cap_item.get("end", 0)) - start_t)
                cap_text = str(cap_item.get("text", "")).strip().upper()
                if c_end > c_start and cap_text:
                    chosen_color = colors[idx % len(colors)]
                    cap_img_array = create_caption_image(cap_text, int(cropped_video.w), chosen_color)
                    txt_clip = (ImageClip(cap_img_array)
                                .set_position(('center', 0.68), relative=True)
                                .set_start(c_start)
                                .set_duration(c_end - c_start))
                    caption_clips.append(txt_clip)
            
            final_video = CompositeVideoClip(caption_clips)
            output_file = "viral_short.mp4"
            final_video.write_videofile(output_file, codec="libx264", audio_codec="aac", fps=24, preset="ultrafast")
            full_video.close()
            final_video.close()
        except Exception as e:
            st.error(f"❌ Video Rendering Error: {str(e)}")
            st.stop()

    st.balloons()
    st.success(f"🔥 Clip Ready! **{clip_data.get('title', 'Viral Short')}** (Viral Score: {clip_data.get('viral_score', 92)}/100)")
    st.video(output_file)
    with open(output_file, "rb") as file:
        st.download_button("⬇️ Download Viral 9:16 Video", file, file_name="viral_short_9x16.mp4", mime="video/mp4")
