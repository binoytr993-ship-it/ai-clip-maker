import streamlit as st
import os
import json
import subprocess
import cv2
import mediapipe as mp
from google import genai
from google.genai import types
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

st.set_page_config(page_title="Free AI Clip Generator", layout="centered")
st.title("🎬 ফ্রী AI ভাইরাল ক্লিপ জেনারেটর (9:16)")

api_key = st.sidebar.text_input("আপনার Gemini API Key দিন:", type="password")

uploaded_file = st.file_uploader("ভিডিও ফাইল আপলোড করুন (MP4/MOV)", type=["mp4", "mov"])
video_path = "input_video.mp4"
audio_path = "temp_audio.mp3"

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
        st.error("দয়া করে বাম পাশের সাইডবারে আপনার Gemini API Key দিন!")
        st.stop()
    if not uploaded_file:
        st.warning("দয়া করে একটি ভিডিও ফাইল আপলোড করুন।")
        st.stop()

    with st.spinner("১/৩: ভিডিও থেকে অডিও আলাদা করা হচ্ছে..."):
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", audio_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    with st.spinner("২/৩: Gemini AI দিয়ে ভাইরাল অংশ ও ক্যাপশন তৈরি হচ্ছে..."):
        try:
            client = genai.Client(api_key=api_key.strip())
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            prompt = """
            এই অডিওটি শুনে সবচেয়ে আকর্ষণীয় ও ভাইরাল হওয়ার মতো ১টি অংশ (১৫ থেকে ৪৫ সেকেন্ডের মধ্যে) খুঁজে বের করো।
            সেই অংশের প্রতি ২-৪ সেকেন্ডের ছোট ছোট ক্যাপশন টাইমস্ট্যাম্পসহ তৈরি করো।
            শুধুমাত্র নিচের JSON ফরম্যাটে উত্তর দাও, অন্য কোনো কথা লিখবে না:
            {
              "start": 10.0,
              "end": 35.0,
              "title": "ক্লিপের শিরোনাম",
              "captions": [
                {"start": 10.0, "end": 13.0, "text": "প্রথম বাক্য"},
                {"start": 13.0, "end": 16.5, "text": "দ্বিতীয় বাক্য"}
              ]
            }
            """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
                    prompt
                ]
            )
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            clip_data = json.loads(clean_json)
        except Exception as e:
            st.error(f"আসল এরর মেসেজ: {str(e)}")
            st.stop()

    with st.spinner("৩/৩: ৯:১৬ ক্রপিং ও ক্যাপশন বসানো হচ্ছে..."):
        start_t = float(clip_data["start"])
        end_t = float(clip_data["end"])
        
        video = VideoFileClip(video_path).subclip(start_t, end_t)
        x1, target_w = get_smart_crop_x(video_path, start_t, end_t)
        cropped_video = video.crop(x1=x1, y1=0, width=target_w, height=video.h)
        
        caption_clips = [cropped_video]
        for cap_item in clip_data.get("captions", []):
            c_start = max(0, float(cap_item["start"]) - start_t)
            c_end = min(end_t - start_t, float(cap_item["end"]) - start_t)
            if c_end > c_start:
                txt = (TextClip(cap_item["text"].strip(), fontsize=40, color='yellow', font='DejaVu-Sans-Bold',
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


                
