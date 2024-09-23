import os
import pickle
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from mimetypes import guess_type as guess_mime_type
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from PIL import Image, ImageTk
from ultralytics import YOLO
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from get_drive import drive_upload
from twilio.rest import Client

import random
# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
OUR_EMAIL = 'your_gmail@gmail.com'  # Replace with your email

def gmail_authenticate():
    """Authenticate and create a service for Gmail API."""
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

def create_message_with_attachment(to, subject, body, file_path):
    """Create an email message with an attachment."""
    message = MIMEMultipart()
    message['to'] = to
    message['from'] = OUR_EMAIL
    message['subject'] = subject

    msg_body = MIMEText(body)
    message.attach(msg_body)

    content_type, encoding = guess_mime_type(file_path)
    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
    main_type, sub_type = content_type.split('/', 1)

    with open(file_path, 'rb') as f:
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(f.read())
    
    filename = os.path.basename(file_path)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)
    
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw_message}

def send_message(service, to, subject, body, file_path):
    """Send an email message with an attachment."""
    message = create_message_with_attachment(to, subject, body, file_path)
    try:
        response = service.users().messages().send(userId='me', body=message).execute()
        print(f'Sent message to {to}. Message Id: {response["id"]}')
    except Exception as error:
        print(f'An error occurred: {error}')

# Load the YOLO and TensorFlow models

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
model = load_model('act.keras')

# Settings
num_keypoints = 33
feature_dim = 3
num_frames = 5
fps = 15
frame_interval = 1/fps

class ActivityRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Activity Recognition")

        # Set canvas size
        self.canvas_width = 640
        self.canvas_height = 480

        # Create GUI components
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.btn_real_time = tk.Button(root, text="Real-Time", command=self.start_real_time)
        self.btn_real_time.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_load_video = tk.Button(root, text="Load Video", command=self.load_video)
        self.btn_load_video.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_exit = tk.Button(root, text="Exit", command=root.quit)
        self.btn_exit.pack(side=tk.RIGHT, padx=10, pady=10)

        self.cap = None
        self.pose_sequence = []
        self.output_video_path = None

    def start_real_time(self):
        self.cap = cv2.VideoCapture(0)
        self.process_stream()

    def set_video_path(self,path):
        self.output_video_path = path
    def get_video_path(self):
        return self.output_video_path

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.set_video_path(self.video_path)
            self.output_video_path = self.video_path  # Store the path to send via email
            self.process_stream()
        else:
            messagebox.showerror("Error", "Failed to load video.")
    def Twilio_Send_Message(self):
        video= self.get_video_path()
        media = drive_upload(video) #Returns the uploaded shareable link
        account_sid = ''  # Your Twilio Account SID
        auth_token = ''
        client = Client(account_sid, auth_token)
        from_whatsapp_number = 'whatsapp:+14155238886'  # Your Twilio sandbox number or verified number
        to_whatsapp_number = ['whatsapp:+91123456789'] #Replace with your number
        video_url = media
        for number in to_whatsapp_number:
            message = client.messages.create(
                body=f"Assault Detected, Check out this video at Location {random.randint(101,151)}",
                from_=from_whatsapp_number,
                to=number,
            )
            message = client.messages.create(
                from_=from_whatsapp_number,
                to=number,
                media_url=video_url
            )
        print(f'Message SID: {message.sid}')
    def process_stream(self):
       
        prev_time = time.time()
        last_label = None

        while self.cap.isOpened():
            current_time = time.time()
            elapsed_time = current_time - prev_time

            if elapsed_time >= frame_interval:
                prev_time = current_time

                ret, frame = self.cap.read()
                if not ret:
                    break

                # Resize the frame to fit the canvas
                frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    pose_frame = []
                    min_x, min_y = 1, 1
                    max_x, max_y = 0, 0

                    for lm in results.pose_landmarks.landmark:
                        min_x = min(min_x, lm.x)
                        min_y = min(min_y, lm.y)
                        max_x = max(max_x, lm.x)
                        max_y = max(max_y, lm.x)
                        pose_frame.extend([lm.x, lm.y, lm.visibility])

                    self.pose_sequence.append(pose_frame)

                    if len(self.pose_sequence) == num_frames:
                        pose_sequence_np = np.expand_dims(np.array(self.pose_sequence), axis=0)
                        prediction = model.predict(pose_sequence_np)
                        predicted_class = np.argmax(prediction)

                        if predicted_class == 0:
                            current_label = 'Assault'
                        elif predicted_class == 1:
                            current_label = 'Fall'
                        elif predicted_class == 2:
                            current_label = 'Kick'
                        elif predicted_class == 3:
                            current_label = 'Kidnap'
                        print(current_label)

                        if current_label != last_label:
                            last_label = current_label

                        self.pose_sequence = []

                if last_label:
                    box_color = (0, 0, 255) if last_label == 'Assault' else (0, 255, 0)
                    cv2.putText(frame_bgr, f'{last_label}', 
                                (int(min_x * frame.shape[1]), int(min_y * frame.shape[0]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2, cv2.LINE_AA)
                    cv2.rectangle(frame_bgr, 
                                  (int(min_x * frame.shape[1]), int(min_y * frame.shape[0])),
                                  (int(max_x * frame.shape[1]), int(max_y * frame.shape[0])),
                                  box_color, 2)
                if last_label == "Assault" or last_label == "Kidnap":
                    self.Twilio_Send_Message()
                # Convert frame to ImageTk format for display in Tkinter canvas
                img = Image.fromarray(frame_bgr)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.root.update_idletasks()

                # Display the frame using OpenCV
                cv2.imshow('Activity Recognition - OpenCV', frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        # Send email after processing the video
        if self.output_video_path:
            self.send_email(self.output_video_path)
    def send_email(self, video_path):
        """Send an email with the loaded video file as an attachment."""
        # Authenticate and send the email
        service = gmail_authenticate()
        send_message(service, "your_gmail@gmail.com", "Assault Detection",
                     "Assault Detected at Location 123", video_path)
        messagebox.showinfo("Email Sent", f"Email sent successfully with the video: {video_path}")
    

if __name__ == "__main__":
    root = tk.Tk()
    app = ActivityRecognitionApp(root)
    root.mainloop()
