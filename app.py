import streamlit as st
import yt_dlp
import os
import json
import cv2
import mediapipe as mp
from faster_whisper import WhisperModel
from google import genai
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

st.set_page_config(page_title="Free AI Clip Generator", layout="centered")
st.title("🎬 ফ্রী AI ভাইরাল ক্লিপ জেনারেটর (9:16)")

api_key = st.sidebar.text_input("আপনার Gemini API Key দিন:", type="password")

input_type = st.radio("ভিডিও ইনপুট মাধ্যম:", ["YouTube URL", "ভিডিও আপলোড"])
video_path = "input_video.mp4"

if input_type == "YouTube URL":
    url = st.text_input("YouTube ভিডিও লিংক পেস্ট করুন:")
else:
    uploaded_file = st.file_uploader("ভিডিও ফাইল আপলোড করুন (MP4)", type=["mp4", "mov"])

def download_youtube(url, output_path):
    ydl_opts = {'format': 'best[ext=mp4]', 'outtmpl': output_path, 'overwrites': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def get_smart_crop_x(video_path, start_t, end_t):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((start_t + (end_t - start_t)/2) * fps))
    ret, frame = cap.read()
    cap.release()
    
    target_width = int(height * (9 / 16))
    center_x = width // 2
    
    if ret:
        results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            face_center = int((bbox.xmin + bbox.width / 2) * width)
            center_x = max(target_width // 2, min(width - target_width // 2, face_center))
            
    return center_x - (target_width // 2), target_width

if st.button("ভাইরাল ক্লিপ তৈরি করুন 🚀"):
    if not api_key:
        st.error("দয়া করে বাম পাশের সাইডবারে Gemini API Key দিন!")
    else:
        with st.spinner("১/৪: ভিডিও প্রস্তুত করা হচ্ছে..."):
            if input_type == "YouTube URL" and url:
                download_youtube(url, video_path)
            elif input_type == "ভিডিও আপলোড" and uploaded_file:
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())
            else:
                st.stop()

        with st.spinner("২/৪: অডিও থেকে টেক্সট (Whisper) তৈরি হচ্ছে..."):
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(video_path, word_timestamps=True)
            segments = list(segments)
            full_transcript = "\n".join([f"[{s.start:.1f}s - {s.end:.1f}s]: {s.text}" for s in segments])

        with st.spinner("৩/৪: AI দিয়ে ভাইরাল অংশ খোঁজা হচ্ছে..."):
            client = genai.Client(api_key=api_key)
            prompt = f"""
            নিচের ভিডিও ট্রান্সক্রিপ্ট থেকে সবচেয়ে আকর্ষণীয় ও ভাইরাল হওয়ার মতো ১টি অংশ (১৫ থেকে ৪৫ সেকেন্ডের মধ্যে) খুঁজে বের করো।
            শুধুমাত্র নিচের JSON ফরম্যাটে উত্তর দাও, অন্য কোনো কথা লিখবে না:
            {{"start": 10.5, "end": 40.0, "title": "ক্লিপের শিরোনাম"}}
            
            ট্রান্সক্রিপ্ট:
            {full_transcript[:6000]}
            """
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            clip_data = json.loads(clean_json)

        with st.spinner("৪/৪: ৯:১৬ ক্রপিং ও ক্যাপশন বসানো হচ্ছে..."):
            start_t = float(clip_data["start"])
            end_t = float(clip_data["end"])
            
            video = VideoFileClip(video_path).subclip(start_t, end_t)
            x1, target_w = get_smart_crop_x(video_path, start_t, end_t)
            cropped_video = video.crop(x1=x1, y1=0, width=target_w, height=video.h)
            
            caption_clips = [cropped_video]
            for seg in segments:
                if seg.end >= start_t and seg.start <= end_t:
                    c_start = max(0, seg.start - start_t)
                    c_end = min(end_t - start_t, seg.end - start_t)
                    txt = (TextClip(seg.text.strip(), fontsize=42, color='yellow', font='DejaVu-Sans-Bold',
                                    stroke_color='black', stroke_width=2, method='caption', size=(target_w*0.85, None))
                           .set_position(('center', 0.72), relative=True)
                           .set_start(c_start)
                           .set_duration(c_end - c_start))
                    caption_clips.append(txt)
            
            final_video = CompositeVideoClip(caption_clips)
            output_file = "viral_short.mp4"
            final_video.write_videofile(output_file, codec="libx264", audio_codec="aac", fps=24)

        st.success(f"তৈরি হয়ে গেছে! শিরোনাম: {clip_data['title']}")
        st.video(output_file)
        with open(output_file, "rb") as file:
            st.download_button("ভিডিও ডাউনলোড করুন ⬇️", file, file_name="viral_clip.mp4")
