import streamlit as st
import os
import json
import subprocess
import cv2
import mediapipe as mp
from google import genai
from google.genai import types
from moviepy.editor import VideoFileClip

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
    .clip-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 15px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🎬 AI Viral Clip Studio (9:16)</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Extract 4 Clean, High-Retention 9:16 Shorts from Any Long Video</div>', unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.markdown("## ⚙️ Studio Settings")
api_key = st.sidebar.text_input("🔑 Enter Gemini API Key:", type="password", placeholder="AQ.Ab8RN6...")
clip_duration = st.sidebar.selectbox("⏱️ Target Clip Length:", ["30 - 55 Seconds (Recommended)", "20 - 40 Seconds", "45 - 60 Seconds"])
st.sidebar.markdown("---")
st.sidebar.info("✨ **Clean Mode:** No captions will be added. Generates 4 distinct 9:16 clips per video.")

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

if st.button("✨ Generate 4 Viral 9:16 Shorts"):
    if not api_key:
        st.error("🚨 Please enter your Gemini API Key in the left sidebar!")
        st.stop()
    if not uploaded_file:
        st.warning("⚠️ Please upload a video file first.")
        st.stop()

    with st.spinner("🎵 Step 1/3: Processing video & compressing audio for full-length AI scan..."):
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "24k", audio_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    with st.spinner("🧠 Step 2/3: AI is scanning the entire video to find the Top 4 viral moments..."):
        try:
            client = genai.Client(api_key=api_key.strip())
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            prompt = f"""
            Listen to the ENTIRE audio from beginning to end. 
            Identify the TOP 4 most engaging, high-retention, viral-worthy segments spread across different parts of the video.
            Each clip MUST have a duration matching: {clip_duration}.
            Ensure the 4 clips do not overlap with each other.
            Provide a catchy English title and a viral score (80-99) for each clip.
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
                            "clips": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "start": {"type": "NUMBER"},
                                        "end": {"type": "NUMBER"},
                                        "title": {"type": "STRING"},
                                        "viral_score": {"type": "INTEGER"}
                                    },
                                    "required": ["start", "end", "title", "viral_score"]
                                }
                            }
                        },
                        "required": ["clips"]
                    }
                )
            )
            data = json.loads(response.text)
            if isinstance(data, dict):
                clips_list = data.get("clips", [])[:4]
            elif isinstance(data, list):
                clips_list = data[:4]
            else:
                clips_list = []
        except Exception as e:
            st.error(f"❌ Gemini API Error: {str(e)}")
            st.stop()

    with st.spinner("🎬 Step 3/3: Smart-cropping all 4 clips to 9:16..."):
        generated_files = []
        try:
            full_video = VideoFileClip(video_path)
            max_dur = full_video.duration
            
            while len(clips_list) < 4:
                idx = len(clips_list)
                fallback_start = min(max_dur - 35, (max_dur / 4) * idx)
                clips_list.append({
                    "start": max(0, fallback_start),
                    "end": min(max_dur, fallback_start + 35),
                    "title": f"Viral Highlight #{idx + 1}",
                    "viral_score": 88
                })

            for i, clip_info in enumerate(clips_list):
                start_t = max(0.0, float(clip_info.get("start", 0)))
                end_t = min(max_dur, float(clip_info.get("end", start_t + 35)))
                if end_t - start_t < 10:
                    end_t = min(max_dur, start_t + 35)
                
                sub = full_video.subclip(start_t, end_t)
                x1, target_w = get_smart_crop_x(video_path, start_t, end_t)
                cropped = sub.crop(x1=x1, y1=0, width=target_w, height=sub.h)
                
                out_name = f"viral_short_{i+1}.mp4"
                cropped.write_videofile(out_name, codec="libx264", audio_codec="aac", fps=24, preset="ultrafast", logger=None)
                generated_files.append({
                    "file": out_name,
                    "title": clip_info.get("title", f"Viral Clip #{i+1}"),
                    "score": clip_info.get("viral_score", 90),
                    "duration": int(end_t - start_t)
                })
            
            full_video.close()
        except Exception as e:
            st.error(f"❌ Video Cutting Error: {str(e)}")
            st.stop()

    st.balloons()
    st.success("🎉 4 Clean 9:16 Viral Shorts Generated Successfully!")
    
    cols = st.columns(2)
    for idx, item in enumerate(generated_files):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="clip-card">
                <h4>🔥 #{idx+1}: {item['title']}</h4>
                <p style="color:#00F2FE; margin:0;">⭐ Viral Score: <b>{item['score']}/100</b> | ⏱️ {item['duration']}s</p>
            </div>
            """, unsafe_allow_html=True)
            st.video(item["file"])
            with open(item["file"], "rb") as f:
                st.download_button(
                    label=f"⬇️ Download Clip #{idx+1}",
                    data=f,
                    file_name=f"viral_short_{idx+1}.mp4",
                    mime="video/mp4",
                    key=f"dl_{idx}"
                )
