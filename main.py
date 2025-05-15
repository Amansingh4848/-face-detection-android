import os
import cv2
import numpy as np
import pickle
import zipfile
from datetime import datetime

from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from kivy.metrics import dp
from kivy.core.window import Window
from kivy.storage.jsonstore import JsonStore
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from kivymd.app import MDApp
from kivymd.uix.list import MDList, OneLineListItem
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.card import MDCard
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.bottomnavigation import MDBottomNavigation
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.chip import MDChip
from kivymd.uix.switch import MDSwitch
from kivymd.uix.boxlayout import MDBoxLayout

# Helper classes for app functionality
class MenuScreen(Screen):
    pass

class FaceStats:
    def __init__(self):
        self.total_detections = 0
        self.successful_recognitions = 0
        self.registration_count = 0
        self.last_detection_time = None
        
    def update_detection(self, recognized=False):
        self.total_detections += 1
        if recognized:
            self.successful_recognitions += 1
        self.last_detection_time = datetime.now()

class AppConfig:
    def __init__(self):
        self.store = JsonStore('face_detection_config.json')
        self.load_defaults()
        
    def load_defaults(self):
        if not self.store.exists('settings'):
            self.store.put('settings',
                theme_style='Dark',
                detection_sensitivity=1.3,
                recognition_threshold=0.5,
                battery_saver=False,
                camera_resolution='720p',
                auto_backup=True
            )
    
    def get_setting(self, key):
        return self.store.get('settings')[key]
        
    def update_setting(self, key, value):
        settings = self.store.get('settings')
        settings[key] = value
        self.store.put('settings', **settings)

class BackupManager:
    def __init__(self, app):
        self.app = app
        self.backup_dir = os.path.join(os.getenv('EXTERNAL_STORAGE', ''), 'FaceDetection/backups')
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            
    def create_backup(self):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(self.backup_dir, f'backup_{timestamp}.zip')
            
            with zipfile.ZipFile(backup_file, 'w') as zf:
                # Backup face data
                for root, dirs, files in os.walk(self.app.faces_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.app.faces_dir)
                        zf.write(file_path, arcname)
                
                # Backup settings
                if os.path.exists('face_detection_config.json'):
                    zf.write('face_detection_config.json')
                    
            return True
        except Exception as e:
            print(f"Backup failed: {str(e)}")
            return False
            
    def restore_backup(self, backup_file):
        try:
            with zipfile.ZipFile(backup_file, 'r') as zf:
                zf.extractall(self.app.faces_dir)
            self.app.load_faces_data()  # Reload face data
            return True
        except Exception as e:
            print(f"Restore failed: {str(e)}")
            return False

# Main application class
class FaceDetectionApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Face Detection System"
        self.theme_cls.material_style = "M3"
        
        # Initialize components
        self.image = None
        self.is_capturing = False
        self.faces_dir = "faces_data"
        self.faces_data = {}
        
        # Initialize additional features
        self.stats = FaceStats()
        self.config = AppConfig()
        self.backup_mgr = BackupManager(self)
        self.file_manager = MDFileManager(
            exit_manager=self.exit_file_manager,
            select_path=self.select_path,
        )
        
        # Apply saved theme
        self.theme_cls.theme_style = self.config.get_setting('theme_style')
        
        # Initialize camera settings
        self.camera_resolution = self.config.get_setting('camera_resolution')
        self.detection_sensitivity = self.config.get_setting('detection_sensitivity')
        self.recognition_threshold = self.config.get_setting('recognition_threshold')
        
        # Create faces directory if not exists
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
        
        self.load_faces_data()
        
        # Initialize face detection
        try:
            cascade_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml"),
                "haarcascade_frontalface_default.xml",
                os.path.join(os.getenv('EXTERNAL_STORAGE', ''), "haarcascade_frontalface_default.xml")
            ]
            
            cascade_path = None
            for path in cascade_paths:
                if os.path.exists(path):
                    cascade_path = path
                    break
            
            if cascade_path is None:
                raise Exception("Could not find face detection model file")
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise Exception(f"Failed to load face detection model from {cascade_path}")
        except Exception as e:
            print(f"Failed to initialize face detection: {str(e)}")

    def build(self):
        # Load KV file
        Builder.load_file('face_detection.kv')
        self.screen = MenuScreen()
        return self.screen

    def toggle_theme(self):
        """Toggle between light and dark theme"""
        self.theme_cls.theme_style = "Light" if self.theme_cls.theme_style == "Dark" else "Dark"
        self.config.update_setting('theme_style', self.theme_cls.theme_style)

    def show_settings(self):
        """Show settings dialog"""
        settings_items = [
            {"text": "Detection Sensitivity", 
             "value": self.detection_sensitivity,
             "hint": "1.0 to 2.0"},
            {"text": "Recognition Threshold", 
             "value": self.recognition_threshold,
             "hint": "0.3 to 0.9"},
            {"text": "Camera Resolution", 
             "value": self.camera_resolution,
             "options": ["480p", "720p", "1080p"]},
            {"text": "Battery Saver Mode", 
             "value": self.config.get_setting('battery_saver'),
             "type": "bool"}
        ]
        
        content = MDBoxLayout(orientation='vertical', spacing=10, padding=10)
        for item in settings_items:
            row = MDBoxLayout(orientation='horizontal')
            row.add_widget(MDLabel(text=item["text"]))
            if "options" in item:
                spinner = MDDropdownMenu(
                    caller=row,
                    items=[{"text": opt} for opt in item["options"]],
                    width_mult=4
                )
                btn = MDRaisedButton(
                    text=str(item["value"]),
                    on_release=lambda x, s=spinner: s.open()
                )
                row.add_widget(btn)
            elif item.get("type") == "bool":
                switch = MDSwitch(active=item["value"])
                row.add_widget(switch)
            else:
                field = MDTextField(
                    text=str(item["value"]),
                    hint_text=item.get("hint", "")
                )
                row.add_widget(field)
            content.add_widget(row)
            
        self.settings_dialog = MDDialog(
            title="Settings",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    on_release=lambda x: self.settings_dialog.dismiss()
                ),
                MDRaisedButton(
                    text="SAVE",
                    on_release=self.save_settings
                ),
            ],
        )
        self.settings_dialog.open()

    def update_stats(self):
        """Update statistics display"""
        if not hasattr(self, 'screen'):
            return
            
        stats = self.stats
        self.screen.ids.total_detections.text = str(stats.total_detections)
        self.screen.ids.recognition_count.text = f"Recognized: {stats.successful_recognitions}"
        self.screen.ids.detection_count.text = f"Detections: {stats.total_detections}"
        
        if stats.total_detections > 0:
            rate = (stats.successful_recognitions / stats.total_detections) * 100
            self.screen.ids.recognition_rate.text = f"{rate:.1f}%"
        
        self.screen.ids.registered_faces.text = str(len(self.faces_data))
        
        if stats.last_detection_time:
            self.screen.ids.last_detection.text = stats.last_detection_time.strftime("%H:%M:%S")

    def show_backup_dialog(self):
        """Show backup/restore dialog"""
        self.backup_dialog = MDDialog(
            title="Backup & Restore",
            text="What would you like to do?",
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    on_release=lambda x: self.backup_dialog.dismiss()
                ),
                MDRaisedButton(
                    text="BACKUP",
                    on_release=self.perform_backup
                ),
                MDRaisedButton(
                    text="RESTORE",
                    on_release=lambda x: self.file_manager.show(self.backup_mgr.backup_dir)
                ),
            ],
        )
        self.backup_dialog.open()

    def perform_backup(self, *args):
        """Create a backup of faces and settings"""
        if self.backup_mgr.create_backup():
            self.show_success_dialog("Backup created successfully")
        else:
            self.show_error_dialog("Failed to create backup")
        self.backup_dialog.dismiss()

    def show_file_manager(self):
        """Show file manager for image import"""
        self.file_manager.show(os.path.expanduser("~"))

    def exit_file_manager(self, *args):
        """Close file manager"""
        self.file_manager.close()

    def select_path(self, path):
        """Handle selected image file"""
        self.exit_file_manager()
        if path.lower().endswith(('.zip')):
            self.backup_mgr.restore_backup(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.process_imported_image(path)

    def process_imported_image(self, image_path):
        """Process imported image for face detection"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, self.detection_sensitivity, 5)
            
            if len(faces) == 0:
                self.show_error_dialog("No face detected in image")
            elif len(faces) > 1:
                self.show_error_dialog("Multiple faces detected in image")
            else:
                x, y, w, h = faces[0]
                face_img = gray[y:y+h, x:x+w]
                self.show_name_dialog(face_img)
        except Exception as e:
            self.show_error_dialog(f"Failed to process image: {str(e)}")

    def load_faces_data(self):
        try:
            data_file = os.path.join(self.faces_dir, "faces_data.pkl")
            if os.path.exists(data_file):
                with open(data_file, "rb") as f:
                    self.faces_data = pickle.load(f)
        except Exception as e:
            print(f"Failed to load faces data: {str(e)}")
            self.faces_data = {}

    def save_faces_data(self):
        try:
            with open(os.path.join(self.faces_dir, "faces_data.pkl"), "wb") as f:
                pickle.dump(self.faces_data, f)
        except Exception as e:
            print(f"Failed to save faces data: {str(e)}")

    def update_faces_list(self):
        self.screen.ids.faces_list.clear_widgets()
        for face_id, data in self.faces_data.items():
            item = OneLineListItem(
                text=f"{data['name']} (ID: {face_id})",
                on_release=lambda x, fid=face_id: self.show_delete_dialog(fid)
            )
            self.screen.ids.faces_list.add_widget(item)

    def toggle_camera(self, instance=None):
        if not self.is_capturing:
            self.capture = cv2.VideoCapture(0)
            if self.capture.isOpened():
                # Set resolution based on settings
                if self.camera_resolution == '1080p':
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                elif self.camera_resolution == '720p':
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                else:  # 480p
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self.is_capturing = True
                self.screen.ids.camera_button.text = "Stop Camera"
                Clock.schedule_interval(self.update_video, 1.0 / 30.0)
        else:
            self.is_capturing = False
            self.screen.ids.camera_button.text = "Start Camera"
            Clock.unschedule(self.update_video)
            self.capture.release()
            self.screen.ids.camera_view.texture = None

    def update_video(self, dt):
        if self.is_capturing:
            ret, frame = self.capture.read()
            if ret:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, self.detection_sensitivity, 5)
                
                # Draw rectangles around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Try to recognize the face
                    face_img = gray[y:y+h, x:x+w]
                    recognized_id = self.recognize_face(face_img)
                    self.stats.update_detection(recognized=bool(recognized_id))
                    
                    if recognized_id:
                        name = self.faces_data[recognized_id]['name']
                        cv2.putText(frame, name, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
                # Update statistics
                self.update_stats()
                
                # Convert frame to texture
                buf = cv2.flip(frame, 0).tostring()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.screen.ids.camera_view.texture = texture

    def recognize_face(self, face_img):
        if not self.faces_data:
            return None
        
        try:
            # Simple face recognition using template matching
            best_match = None
            highest_similarity = -1
            
            for face_id, data in self.faces_data.items():
                template = cv2.imread(data['face_path'], cv2.IMREAD_GRAYSCALE)
                if template is None:
                    continue
                
                if template.shape != face_img.shape:
                    template = cv2.resize(template, face_img.shape[::-1])
                
                # Calculate similarity using normalized correlation coefficient
                result = cv2.matchTemplate(face_img, template, cv2.TM_CCOEFF_NORMED)
                similarity = result.max()
                
                if similarity > highest_similarity and similarity > self.recognition_threshold:
                    highest_similarity = similarity
                    best_match = face_id
            
            return best_match
        except Exception:
            return None

    def show_error_dialog(self, text):
        dialog = MDDialog(
            title="Error",
            text=text,
            buttons=[
                MDFlatButton(
                    text="OK",
                    on_release=lambda x: dialog.dismiss()
                )
            ],
        )
        dialog.open()

    def show_success_dialog(self, text):
        dialog = MDDialog(
            title="Success",
            text=text,
            buttons=[
                MDFlatButton(
                    text="OK",
                    on_release=lambda x: dialog.dismiss()
                )
            ],
        )
        dialog.open()

if __name__ == '__main__':
    FaceDetectionApp().run()
