from flask import Flask, request, render_template, send_file, redirect, url_for, flash
import os
import subprocess
import cv2
from werkzeug.utils import secure_filename
from gtts import gTTS
from moviepy.editor import VideoFileClip
import tempfile
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import datetime
from flask_mail import Mail, Message
import requests
import base64
import torch
from torchvision import models, transforms
from PIL import Image
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from IPython.display import display
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline 
from torchvision.transforms import ToTensor, Normalize, ConvertImageDtype
import ipywidgets as widgets
from IPython.display import clear_output
import moviepy.editor as mp
import time
from IPython.display import display, clear_output, Audio
from gtts import gTTS
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline 
from torchvision.transforms import ToTensor, Normalize, ConvertImageDtype
import cv2
import os
from moviepy.editor import VideoFileClip, AudioFileClip
import os
import glob

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/newdatabase'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'ethicalgan@gmail.com'
app.config['MAIL_PASSWORD'] = 'rehg hjfx tauh zrof'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.secret_key = "supersecretkey"

mail = Mail(app)
mongo = PyMongo(app)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
WAV2LIP_DIR = "Wav2Lip"
WEIGHTS_DIR = 'weights'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

@app.route('/')
def home():
    return render_template('login.html')



@app.route('/login', methods=['POST'])
def login():
    email = request.form['email'].strip()
    password = request.form['password']
    print(email)
    print(password)
    user = mongo.db.users.find_one({'email': email})
    if user and check_password_hash(user['password'], password):
        flash('Login successful!')
        return redirect(url_for('wav2lip'))
    else:
        flash('Invalid username or password.')
        return redirect(url_for('home'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password']
        print(email)
        print(password)
        # Check if the username already exists in the database
        existing_user = mongo.db.users.find_one({'email': email})
        
        if existing_user:
            flash('Username already exists. Please choose a different username.')
            return redirect(url_for('signup'))
        # If username does not exist, proceed to create a new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        mongo.db.users.insert_one({'email': email, 'password': hashed_password})
        flash('Signup successful! Please log in.')
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email'].strip()  
        user = mongo.db.users.find_one({'email': email})  
        
        if user:
            # Generate a token
            token = str(uuid.uuid4())
            mongo.db.password_reset_tokens.insert_one({
                'email': email,  # Store the token with the email
                'token': token,
            })

            # Generate the reset URL
            reset_url = url_for('reset_password', token=token, _external=True)

            # Create a message to send the email
            msg = Message('Password Reset Request',
                          sender='ethicalgan@gmail.com',
                          recipients=[email])
            msg.body = f'Click the following link to reset your password: {reset_url}'

            try:
                mail.send(msg)  # Send the email
                flash('A password reset link has been sent to your email.', 'info')
            except Exception as e:
                flash(f'Error sending email: {str(e)}', 'danger')
        else:
            flash('Email not found. Please check the email address.', 'danger')

        return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html')

@app.route('/logout')
def logout():
    flash('You have been logged out.')
    return redirect(url_for('home'))

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if request.method == 'POST':
        new_password = request.form['password']
        reset_token = mongo.db.password_reset_tokens.find_one({'token': token})
        if reset_token and reset_token['expires_at'] > datetime.datetime.now():
            username = reset_token['username']
            hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
            mongo.db.users.update_one({'username': username}, {'$set': {'password': hashed_password}})
            mongo.db.password_reset_tokens.delete_one({'token': token})
            flash('Password has been reset successfully! You can now log in.')
            return redirect(url_for('home'))
        else:
            flash('Invalid or expired token.')
    return render_template('reset_password.html', token=token)

def get_video_resolution(video_path):
    """Get the resolution of a video."""
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    return (width, height)

def resize_video(video_path, new_resolution):
    """Resize a video to the given resolution."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp_resized_video.mp4")
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    width, height = new_resolution
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    while True:
        success, frame = video.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (width, height))
        writer.write(resized_frame)
    
    video.release()
    writer.release()
    shutil.move(temp_path, video_path)
    shutil.rmtree(temp_dir)
    return video_path

# Model URLs
model_urls = {
    'realesr-general-x4v3.pth': "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    'GFPGANv1.4.pth': "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    'RestoreFormer.pth': "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
    'CodeFormer.pth': "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth",
}

def download_file(url, filename):
    """Download a file from URL and save it"""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

# Download model weights if they don't exist
for filename, url in model_urls.items():
    file_path = os.path.join(WEIGHTS_DIR, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        download_file(url, file_path)

# Initialize models
realesrgan_model_path = os.path.join(WEIGHTS_DIR, 'realesr-general-x4v3.pth')
gfpgan_model_path = os.path.join(WEIGHTS_DIR, 'GFPGANv1.4.pth')

# Initialize RealESRGAN
sr_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
realesrganer = RealESRGANer(scale=4, model_path=realesrgan_model_path, model=sr_model, tile=0, tile_pad=10, pre_pad=0, half=True)

# Initialize GFPGAN
face_enhancer = GFPGANer(model_path=gfpgan_model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=realesrganer)

def get_video_resolution(video_path):
    """Get video resolution"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (width, height)

def resize_video(video_path, new_resolution):
    """Function to resize a video and handle file replacement"""
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    temp_path = os.path.join(temp_dir, "temp_resized_video.mp4")
    backup_path = video_path + ".bak"  # Backup path for original file
    
    try:
        # Open the original video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error opening video file: {video_path}")
            return None
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
        fps = video.get(cv2.CAP_PROP_FPS)
        width, height = new_resolution
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        while True:
            success, frame = video.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, (width, height))
            writer.write(resized_frame)
        
        video.release()
        writer.release()
        
        # Try to rename the original file
        for attempt in range(5):  # Retry up to 5 times
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)  # Ensure backup file is not present
                os.rename(video_path, backup_path)
                break
            except PermissionError as e:
                print(f"Attempt {attempt+1}: Unable to rename the original file '{video_path}'. Error: {e}")
                time.sleep(1)  # Wait a moment before retrying
        
        # Replace the original file with the resized video
        try:
            shutil.move(temp_path, video_path)
        except PermissionError as e:
            print(f"Unable to replace the original file '{video_path}' with the resized video. Error: {e}")
            return None
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        return video_path
    except Exception as e:
        print(f"Error during video processing: {e}")
        return None

def extract_frames(video_path, output_dir):
    """Extract frames from video"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.png"), frame)
    cap.release()
    return frame_count

def enhance_faces(image_path, output_path):
    """Enhance faces in image using GFPGAN"""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    _, _, img_enhanced = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    cv2.imwrite(output_path, img_enhanced)
    return img_enhanced

def reassemble_video(frame_dir, output_video_path, fps):
    """Reassemble video from frames"""
    frame_list = sorted([os.path.join(frame_dir, img) for img in os.listdir(frame_dir)])
    frame = cv2.imread(frame_list[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        video.write(frame)
    video.release()

def add_audio_to_video(video_path, audio_path, output_path):
    """Add audio to video"""
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")


@app.route("/wav2lip", methods=["GET", "POST"])
def wav2lip():
    if request.method == "POST":
        video_file = request.files.get("video")
        audio_file = request.files.get("audio")
        tts_text = request.form.get("tts_text")

        if not video_file:
            flash("Please upload a video file.", "error")
            return redirect(url_for("index"))

        video_filename = secure_filename(video_file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video_file.save(video_path)

        # Check video duration
        video_duration = VideoFileClip(video_path).duration
        if video_duration > 60:
            flash("Video duration exceeds 60 seconds. Please upload a shorter video.", "error")
            os.remove(video_path)
            return redirect(url_for("wav2lip"))

        # Resize video if resolution is too high
        video_resolution = get_video_resolution(video_path)
        if video_resolution[0] > 1280 or video_resolution[1] > 720:
            resize_video(video_path, (1280, 720))

        if audio_file:
            audio_filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
            audio_file.save(audio_path)
        elif tts_text:
            tts = gTTS(tts_text)
            audio_path = os.path.join(UPLOAD_FOLDER, "tts_audio.wav")
            tts.save(audio_path)
        else:
            flash("Please upload an audio file or provide text for TTS.", "error")
            return redirect(url_for("wav2lip"))

        # Generate lip-synced video
        result_path = os.path.join(RESULT_FOLDER, "lip_synced_video.mp4")
        try:
            subprocess.run(
                [
                    "python", f"{WAV2LIP_DIR}/inference.py",
                    "--checkpoint_path", f"{WAV2LIP_DIR}/checkpoints/wav2lip_gan.pth",
                    "--face", video_path,
                    "--audio", audio_path,
                    "--outfile", result_path,
                ],
                check=True
            )
            # Generate lip-synced video
            initial_result_path = os.path.join(RESULT_FOLDER, "lip_synced_video.mp4")

            if os.path.exists(initial_result_path):
                # Get FPS
                print("Extracting FPS of the video...")
                ffprobe_command = [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate",
                    "-of", "default=noprint_wrappers=1:nokey=1", result_path
                ]
                fps_process = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True)
                fps_fraction = fps_process.stdout.strip()
                fps = float(eval(fps_fraction)) if '/' in fps_fraction else float(fps_fraction)
                print(f"FPS of the current session is: {fps}")

                # Extract and enhance frames
                frames_dir = os.path.join(RESULT_FOLDER, "frames")
                enhanced_frames_dir = os.path.join(RESULT_FOLDER, "enhanced_frames")
                os.makedirs(frames_dir, exist_ok=True)
                os.makedirs(enhanced_frames_dir, exist_ok=True)

                # Extract frames
                extract_frames(initial_result_path, frames_dir)

                # Enhance each frame
                for frame in sorted(os.listdir(frames_dir)):
                    frame_path = os.path.join(frames_dir, frame)
                    enhanced_frame_path = os.path.join(enhanced_frames_dir, frame)
                    enhance_faces(frame_path, enhanced_frame_path)

                # Reassemble video
                enhanced_video_path = os.path.join(RESULT_FOLDER, "enhanced_video.mp4")
                reassemble_video(enhanced_frames_dir, enhanced_video_path, fps)

                # Add audio to final video
                final_video_path = os.path.join(RESULT_FOLDER, "final_video.mp4")
                add_audio_to_video(enhanced_video_path, audio_path, final_video_path)

                # Clean up temporary files
                for temp_dir in [frames_dir, enhanced_frames_dir]:
                    for file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file))
                    os.rmdir(temp_dir)

                return render_template("result.html", result_video=final_video_path)

        except subprocess.CalledProcessError as e:
            flash("Error during video processing. Check the server logs.", "error")
            return redirect(url_for("wav2lip"))

    return render_template("wav2lip.html")

@app.route("/download/<path:filename>")
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run()
