#!/usr/bin/env python3
"""
Integrated Face Recognition and Hand Landmark Tracking System
Raspberry Pi 5 optimized with shared camera and minimal resource usage
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pickle
import os
import time
import threading
import queue
from picamera2 import Picamera2
import mediapipe as mp
import sys
import tkinter as tk
from tkinter import ttk
from pynput import mouse
from pynput.mouse import Button, Controller, Listener

# Import hand tracking modules
sys.path.append('handMini2')
from utils import calc_landmark
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from ocr_bridge import recognize_text

# Configuration
CONFIDENCE_THRESHOLD = 0.15
SIMILARITY_THRESHOLD = 0.6
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
DETECTION_MODEL_PATH = "face/Lightweight-Face-Detection.tflite"
EMBEDDING_MODEL_PATH = "face/MobileFaceNet_9925_9680.tflite"
FACE_DATABASE_FILENAME = "face/pi_face_database.pkl"
FACE_RECOGNITION_DURATION = 3.0  # 3 seconds for face recognition

class SharedCamera:
    """Shared camera manager for both face and hand tracking"""
    def __init__(self):
        self.picam2 = None
        self.is_initialized = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = False
        self.camera_thread = None
    
    def initialize(self):
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
                controls={"FrameDurationLimits": (33333, 33333)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            print("✓ Shared camera initialized successfully")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"✗ Camera initialization error: {e}")
            return False
    
    def start_camera_stream(self):
        """Start continuous camera stream"""
        if not self.is_initialized:
            return False
        
        self.is_running = True
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        return True
    
    def _camera_loop(self):
        """Continuous camera capture loop"""
        try:
            while self.is_running:
                try:
                    frame = self.picam2.capture_array()
                    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.flip(frame, 1)  # Mirror for better UX
                    
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    
                    time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                except Exception as e:
                    print(f"✗ Frame capture error: {e}")
                    time.sleep(0.1)  # Longer delay on error
        except Exception as e:
            print(f"✗ Camera loop error: {e}")
    
    def get_current_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def stop_camera_stream(self):
        """Stop camera stream"""
        self.is_running = False
        if self.camera_thread:
            try:
                self.camera_thread.join(timeout=2.0)
            except:
                pass
    
    def close(self):
        self.stop_camera_stream()
        if self.picam2 is not None:
            self.picam2.close()

class FaceRecognitionManager:
    """Face recognition system manager"""
    def __init__(self, camera):
        self.camera = camera
        self.models = {}
        self.is_loaded = False
        self.detection_input = np.zeros((1, 480, 640, 1), dtype=np.float32)
        self.embedding_input = np.zeros((1, 112, 112, 3), dtype=np.float32)
        self.face_database = {}
        self.recognition_start_time = None
        self.current_recognized_face = None
        self.load_database()
    
    def load_models(self):
        try:
            # Load detection model
            if not os.path.exists(DETECTION_MODEL_PATH):
                print("✗ Face detection model not found")
                return False
            
            detection_interpreter = tflite.Interpreter(model_path=DETECTION_MODEL_PATH)
            detection_interpreter.allocate_tensors()
            self.models['detection'] = {
                'interpreter': detection_interpreter,
                'input': detection_interpreter.get_input_details(),
                'output': detection_interpreter.get_output_details()
            }
            print("✓ Face detection model loaded")
            
            # Load embedding model
            if not os.path.exists(EMBEDDING_MODEL_PATH):
                print("✗ Face embedding model not found")
                return False
            
            embedding_interpreter = tflite.Interpreter(model_path=EMBEDDING_MODEL_PATH)
            embedding_interpreter.allocate_tensors()
            self.models['embedding'] = {
                'interpreter': embedding_interpreter,
                'input': embedding_interpreter.get_input_details(),
                'output': embedding_interpreter.get_output_details()
            }
            print("✓ Face embedding model loaded")
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"✗ Face model loading error: {e}")
            return False
    
    def load_database(self):
        if os.path.exists(FACE_DATABASE_FILENAME):
            try:
                with open(FACE_DATABASE_FILENAME, 'rb') as f:
                    self.face_database = pickle.load(f)
                print(f"✓ Face database loaded: {len(self.face_database)} faces")
            except Exception as e:
                print(f"✗ Error loading face database: {e}")
                self.face_database = {}
        else:
            print("✓ No existing face database found")
            self.face_database = {}
    
    def save_database(self):
        try:
            with open(FACE_DATABASE_FILENAME, 'wb') as f:
                pickle.dump(self.face_database, f)
            print("✓ Face database saved")
            return True
        except Exception as e:
            print(f"✗ Error saving face database: {e}")
            return False
    
    def detect_face(self, image):
        if not self.is_loaded or 'detection' not in self.models:
            return None
        
        detection_info = self.models['detection']
        interpreter = detection_info['interpreter']
        input_details = detection_info['input']
        output_details = detection_info['output']
        
        resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        self.detection_input[0, :, :, 0] = gray.astype(np.float32) / 255.0
        
        interpreter.set_tensor(input_details[0]['index'], self.detection_input)
        interpreter.invoke()
        
        heatmap = interpreter.get_tensor(output_details[0]['index'])
        bbox = interpreter.get_tensor(output_details[1]['index'])
        
        H0, W0 = image.shape[:2]
        H_in, W_in = 480, 640
        G_H, G_W = 60, 80
        STRIDE_X, STRIDE_Y = W_in / G_W, H_in / G_H
        
        heatmap_2d = heatmap[0, :, :, 0]
        ys, xs = np.where(heatmap_2d > CONFIDENCE_THRESHOLD)
        
        if ys.size == 0:
            return None
        
        scores = heatmap_2d[ys, xs]
        cx = (xs + 0.5) * STRIDE_X
        cy = (ys + 0.5) * STRIDE_Y
        
        boxes = []
        for i in range(len(ys)):
            y, x = ys[i], xs[i]
            dy1, dx1, dy2, dx2 = bbox[0, y, x, :]
            x1 = cx[i] - dx1 * STRIDE_X
            y1 = cy[i] - dy1 * STRIDE_Y
            x2 = cx[i] + dx2 * STRIDE_X
            y2 = cy[i] + dy2 * STRIDE_Y
            boxes.append([x1, y1, x2, y2])
        
        if not boxes:
            return None
        
        boxes_pix = np.array(boxes)
        boxes_pix[:, [0,2]] *= W0 / W_in
        boxes_pix[:, [1,3]] *= H0 / H_in
        
        bboxes_for_nms = [[x1,y1,x2-x1,y2-y1] for x1,y1,x2,y2 in boxes_pix]
        idxs = cv2.dnn.NMSBoxes(bboxes_for_nms, scores.tolist(), CONFIDENCE_THRESHOLD, 0.3)
        
        if len(idxs) > 0:
            best_idx = idxs[np.argmax(scores[idxs])]
            x1, y1, x2, y2 = boxes_pix[best_idx]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            if x1 >= 0 and y1 >= 0 and x2 < W0 and y2 < H0 and x2 > x1 and y2 > y1:
                w, h = x2 - x1, y2 - y1
                if 30 <= w <= 400 and 30 <= h <= 400:
                    return (x1, y1, w, h, scores[best_idx])
        return None
    
    def get_face_embedding(self, face_img):
        if not self.is_loaded or 'embedding' not in self.models:
            return None
        
        embedding_info = self.models['embedding']
        interpreter = embedding_info['interpreter']
        input_details = embedding_info['input']
        output_details = embedding_info['output']
        
        face_resized = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        self.embedding_input[0] = face_rgb.astype(np.float32) / 255.0
        
        interpreter.set_tensor(input_details[0]['index'], self.embedding_input)
        interpreter.invoke()
        embedding = interpreter.get_tensor(output_details[0]['index']).flatten()
        
        return embedding
    
    def compare_faces(self, embedding1, embedding2):
        if embedding1 is None or embedding2 is None:
            return 0.0, False
        
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0, False
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return similarity, similarity > SIMILARITY_THRESHOLD
    
    def find_best_match(self, embedding):
        if embedding is None or not self.face_database:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        for name, registered_embedding in self.face_database.items():
            similarity, _ = self.compare_faces(embedding, registered_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        return best_match, best_similarity
    
    def register_face(self, name, embedding):
        if not name or embedding is None:
            return False
        try:
            self.face_database[name] = embedding
            print(f"✓ '{name}' face registered successfully")
            return True
        except Exception as e:
            print(f"✗ Error registering face: {e}")
            return False
    
    def process_frame(self, frame):
        if frame is None:
            return None, None, None
        
        face_result = self.detect_face(frame)
        if not face_result:
            # Reset recognition timer if no face detected
            self.recognition_start_time = None
            self.current_recognized_face = None
            return frame, None, None
        
        x, y, w, h, _ = face_result
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            current_embedding = self.get_face_embedding(face_roi)
            if current_embedding is not None:
                best_match, best_similarity = self.find_best_match(current_embedding)
                
                if best_match and best_similarity > SIMILARITY_THRESHOLD:
                    # Check if this is the same face we've been tracking
                    if self.current_recognized_face == best_match:
                        if self.recognition_start_time is None:
                            self.recognition_start_time = time.time()
                        
                        # Check if we've been recognizing this face for 3 seconds
                        if time.time() - self.recognition_start_time >= FACE_RECOGNITION_DURATION:
                            return frame, face_result, (best_match, best_similarity, True)  # True = login ready
                        else:
                            remaining_time = FACE_RECOGNITION_DURATION - (time.time() - self.recognition_start_time)
                            return frame, face_result, (best_match, best_similarity, False, remaining_time)
                    else:
                        # New face detected, reset timer
                        self.current_recognized_face = best_match
                        self.recognition_start_time = time.time()
                        return frame, face_result, (best_match, best_similarity, False, FACE_RECOGNITION_DURATION)
                else:
                    # Reset recognition timer if face doesn't match
                    self.recognition_start_time = None
                    self.current_recognized_face = None
                    return frame, face_result, (None, 0, False)
            else:
                self.recognition_start_time = None
                self.current_recognized_face = None
                return frame, face_result, (None, 0, False)
        except Exception as e:
            print(f"Face recognition error: {e}")
            self.recognition_start_time = None
            self.current_recognized_face = None
            return frame, face_result, (None, 0, False)

# 기존 코드에서 HandTrackingManager 클래스 시작 부분을 찾아서 아래 코드로 전체 교체합니다.

# 기존 코드에서 HandTrackingManager 클래스 전체를 찾아 아래 코드로 교체합니다.

# 기존 코드에서 HandTrackingManager 클래스 전체를 찾아 아래 코드로 교체합니다.

class HandTrackingManager:
    """Hand landmark tracking system manager with screen capture."""
    def __init__(self, camera, tkinter_queue=None):
        self.camera = camera
        self.tkinter_queue = tkinter_queue
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = None
        self.keypoint_classifier = None
        self.is_initialized = False
        
        # Mode and state variables
        self.current_mode = "Mouse Control" # "Mouse Control", "OCR Capture", "Screen Capture"
        self.mode_toggle_cooldown = 0
        self.awaiting_ocr_confirmation = False
        self.awaiting_capture_confirmation = False

        # Mouse control variables
        self.mouse_controller = Controller()
        self.screen_width, self.screen_height = self.get_screen_size()
        self.last_finger_pos = None
        self.finger_stable_start_time = None
        self.finger_stable_threshold = 20  # pixels
        self.dwell_click_duration = 1.5  # 클릭 딜레이를 1.5초로 약간 줄임

        # Capture variables
        self.capture_points = [] # For OCR
        self.screen_capture_points = [] # For Screen Capture
        
    def initialize(self):
        try:
            original_dir = os.getcwd()
            if not os.path.exists('handMini2'):
                print("✗ Error: 'handMini2' directory not found.")
                return False
            os.chdir('handMini2')
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.8
            )
            self.keypoint_classifier = KeyPointClassifier()
            os.chdir(original_dir)
            self.is_initialized = True
            print("✓ Hand tracking models initialized")
            return True
        except Exception as e:
            if 'original_dir' in locals(): os.chdir(original_dir)
            print(f"✗ Hand tracking initialization error: {e}")
            return False
    
    def get_screen_size(self):
        try:
            root = tk.Tk()
            root.withdraw()
            width, height = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return width, height
        except:
            print("Warning: Could not get screen size via Tkinter. Using default 1920x1080.")
            return 1920, 1080
    
    def map_finger_to_screen(self, finger_x, finger_y, frame_width, frame_height):
        screen_x = int((finger_x / frame_width) * self.screen_width)
        screen_y = int((finger_y / frame_height) * self.screen_height)
        return max(0, min(screen_x, self.screen_width - 1)), max(0, min(screen_y, self.screen_height - 1))

    def process_frame(self, frame):
        if not self.is_initialized or frame is None:
            return frame, None, None
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                landmark_list = calc_landmark.calc_landmark(frame, hand_landmarks)
                pre_processed = calc_landmark.pre_process_landmark(landmark_list)
                gesture_id = self.keypoint_classifier(pre_processed)
                gesture = self.keypoint_classifier.labels[gesture_id]

                self.handle_gestures(gesture, hand_landmarks, frame.shape)
        
        self.draw_ui(frame)
        self.mode_toggle_cooldown = max(0, self.mode_toggle_cooldown - 1)
        return frame, gesture, results.multi_hand_landmarks

    def handle_gestures(self, gesture, hand_landmarks, frame_shape):
        if gesture == "Open" and self.mode_toggle_cooldown == 0:
            modes = ["Mouse Control", "Screen Capture", "OCR Capture"]
            try:
                current_index = modes.index(self.current_mode)
                next_index = (current_index + 1) % len(modes)
                self.current_mode = modes[next_index]
            except ValueError:
                self.current_mode = "Mouse Control"
            
            print(f"Mode changed to: {self.current_mode}")
            self.reset_mode_state()
            self.mode_toggle_cooldown = 30 

        elif gesture == "Pointer":
            tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_x, finger_y = int(tip.x * frame_shape[1]), int(tip.y * frame_shape[0])
            screen_pos = self.map_finger_to_screen(finger_x, finger_y, frame_shape[1], frame_shape[0])

            self.mouse_controller.position = screen_pos
            
            if self.current_mode == "Mouse Control":
                self.handle_dwell_click(screen_pos)
            elif self.current_mode == "Screen Capture":
                self.handle_screen_capture_pointing(screen_pos)
        
        elif gesture == "Close" and self.awaiting_capture_confirmation:
            self.perform_screen_capture()
            self.reset_mode_state()

    def handle_dwell_click(self, current_pos):
        # <<--- FIX 2: 클릭 로직 구현 ---
        distance = np.linalg.norm(np.array(current_pos) - np.array(self.last_finger_pos)) if self.last_finger_pos else float('inf')

        if distance <= self.finger_stable_threshold:
            if self.finger_stable_start_time is None:
                self.finger_stable_start_time = time.time()
            elif time.time() - self.finger_stable_start_time >= self.dwell_click_duration:
                self.mouse_controller.click(Button.left)
                print(f"✓ Dwell click performed at {current_pos}")
                self.finger_stable_start_time = None # Reset timer after click
        else:
            self.finger_stable_start_time = None
        
        self.last_finger_pos = current_pos

    def handle_screen_capture_pointing(self, current_pos):
        if len(self.screen_capture_points) == 1:
            self.update_screen_box_drawing(self.screen_capture_points[0], current_pos)
            
        distance = np.linalg.norm(np.array(current_pos) - np.array(self.last_finger_pos)) if self.last_finger_pos else float('inf')

        if distance <= self.finger_stable_threshold:
            if self.finger_stable_start_time is None:
                self.finger_stable_start_time = time.time()
            elif time.time() - self.finger_stable_start_time >= self.dwell_click_duration:
                if len(self.screen_capture_points) == 0:
                    self.screen_capture_points.append(current_pos)
                    print(f"Capture start point set: {current_pos}")
                    self.start_screen_box_drawing()
                elif len(self.screen_capture_points) == 1:
                    self.screen_capture_points.append(current_pos)
                    print(f"Capture end point set: {current_pos}")
                    self.awaiting_capture_confirmation = True
                self.finger_stable_start_time = None
        else:
            self.finger_stable_start_time = None
        
        self.last_finger_pos = current_pos

    def perform_screen_capture(self):
        if len(self.screen_capture_points) != 2: return

        x1, y1 = self.screen_capture_points[0]
        x2, y2 = self.screen_capture_points[1]
        
        left, top, width, height = min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)
        if width < 10 or height < 10:
            print("✗ Capture region too small, cancelled.")
            return

        try:
            import pyautogui
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screen_capture.png"
            screenshot.save(filename)
            print(f"✓ Screen capture saved as: {filename}")
        except Exception as e:
            print(f"✗ Screen capture failed: {e}")
        finally:
            self.stop_screen_box_drawing()

    def start_screen_box_drawing(self):
        if self.tkinter_queue:
            self.tkinter_queue.put(('start_box_drawing', self.screen_width, self.screen_height))

    def update_screen_box_drawing(self, p1, p2):
        if self.tkinter_queue:
            x1, y1 = p1
            x2, y2 = p2
            self.tkinter_queue.put(('update_box_drawing', x1, y1, x2, y2))
            
    def stop_screen_box_drawing(self):
        if self.tkinter_queue:
            self.tkinter_queue.put(('stop_box_drawing',))
    
    def reset_mode_state(self):
        self.capture_points, self.screen_capture_points = [], []
        self.awaiting_ocr_confirmation, self.awaiting_capture_confirmation = False, False
        self.last_finger_pos, self.finger_stable_start_time = None, None
        self.stop_screen_box_drawing()
        
    def handle_key(self, key):
        if key == ord('r'):
            print("Restarting current mode state.")
            self.reset_mode_state()
            return False
        return True
        
    def draw_ui(self, frame):
        cv2.putText(frame, f"Mode: {self.current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        instruction = ""
        if self.current_mode == "Screen Capture":
            if self.awaiting_capture_confirmation:
                instruction = "'Close' gesture: CAPTURE | 'r': RESTART"
            elif len(self.screen_capture_points) == 0:
                instruction = "Dwell with 'Pointer' to set START point"
            else:
                instruction = "Dwell with 'Pointer' to set END point"
        elif self.current_mode == "Mouse Control":
            instruction = "Dwell with 'Pointer' for 1.5s to CLICK"

        if instruction:
            cv2.putText(frame, instruction, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
# 기존 코드에서 IntegratedGUI 클래스 전체를 찾아 아래 코드로 교체합니다.

# 기존 코드에서 IntegratedGUI 클래스 전체를 찾아 아래 코드로 교체합니다.

# 기존 코드에서 IntegratedGUI 클래스 전체를 찾아 아래 코드로 교체합니다.

class IntegratedGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face & Hand Tracking System")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        self.tkinter_queue = queue.Queue()
        self.camera = SharedCamera()
        self.face_manager = FaceRecognitionManager(self.camera)
        self.hand_manager = HandTrackingManager(self.camera, self.tkinter_queue)
        
        # --- 오버레이 윈도우 관련 변수 ---
        self.overlay_window = None
        self.overlay_canvas = None
        self.transparent_color = '#abcdef' 
        # --- self.selection_rect_id 변수 삭제 ---

        self.is_logged_in = False
        self.current_user = None
        self.is_running = False
        self.current_mode = "idle"
        
        self.processing_thread = None
        self.setup_gui()
        self.process_tkinter_queue()
    
    def process_tkinter_queue(self):
        try:
            while not self.tkinter_queue.empty():
                command = self.tkinter_queue.get_nowait()
                cmd_type = command[0]
                
                if cmd_type == 'start_box_drawing':
                    self._start_screen_box_drawing(command[1], command[2])
                elif cmd_type == 'update_box_drawing':
                    self._update_screen_box_drawing(*command[1:])
                elif cmd_type == 'stop_box_drawing':
                    self._stop_screen_box_drawing()
        except queue.Empty:
            pass
        self.root.after(50, self.process_tkinter_queue)
    
    def _start_screen_box_drawing(self, screen_width, screen_height):
        if self.overlay_window: return
        try:
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.attributes('-topmost', True)
            self.overlay_window.overrideredirect(True)
            self.overlay_window.geometry(f"{screen_width}x{screen_height}+0+0")
            self.overlay_window.wm_attributes("-transparentcolor", self.transparent_color)
            
            self.overlay_canvas = tk.Canvas(self.overlay_window, width=screen_width, height=screen_height,
                                          bg=self.transparent_color, highlightthickness=0)
            self.overlay_canvas.pack()
            print("✓ Screen box drawing overlay started")
        except Exception as e:
            print(f"✗ Error starting screen box drawing: {e}")
            if self.overlay_window: self.overlay_window.destroy()
            self.overlay_window = None

    # <<--- FIX: 매번 사각형을 지우고 새로 그리는 방식으로 변경 ---
    def _update_screen_box_drawing(self, x1, y1, x2, y2):
        if not self.overlay_canvas: return
        try:
            # 캔버스의 모든 것을 지움
            self.overlay_canvas.delete("all")
            # 새로운 좌표로 사각형을 다시 그림
            self.overlay_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=3)
        except Exception as e:
            print(f"✗ Error updating screen box drawing: {e}")

    def _stop_screen_box_drawing(self):
        if self.overlay_window:
            try:
                self.overlay_window.destroy()
                print("✓ Screen box drawing overlay stopped")
            except Exception: pass
            finally:
                self.overlay_window = None
                self.overlay_canvas = None
    
    # --- 나머지 IntegratedGUI의 함수들은 이전과 동일합니다 ---
    # ... (setup_gui, start_face_recognition 등 나머지 함수들은 그대로 두시면 됩니다)
    def setup_gui(self):
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        title_label = tk.Label(main_frame, text="Face Recognition & Hand Tracking System", font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=(0, 20))
        self.status_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        self.status_frame.pack(fill=tk.X, pady=(0, 20))
        self.status_label = tk.Label(self.status_frame, text="Status: Ready", font=('Arial', 12), fg='white', bg='#34495e')
        self.status_label.pack(pady=10)
        control_frame = tk.Frame(main_frame, bg='#2c3e50')
        control_frame.pack(fill=tk.X, pady=(0, 20))
        self.login_button = tk.Button(control_frame, text="Start Face Recognition", command=self.start_face_recognition, font=('Arial', 12), bg='#3498db', fg='white', relief=tk.RAISED, bd=3, padx=20, pady=10)
        self.login_button.pack(side=tk.LEFT, padx=(0, 10))
        self.logout_button = tk.Button(control_frame, text="Logout", command=self.logout, font=('Arial', 12), bg='#e74c3c', fg='white', relief=tk.RAISED, bd=3, padx=20, pady=10)
        self.user_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        self.user_info_label = tk.Label(self.user_frame, text="Not logged in", font=('Arial', 12), fg='white', bg='#34495e')
        self.user_info_label.pack(pady=10)

    def start_face_recognition(self):
        if self.is_running: return
        self.update_status("Initializing camera...")
        if not self.camera.initialize():
            self.update_status("Camera initialization failed")
            return
        self.update_status("Loading face models...")
        if not self.face_manager.load_models():
            self.update_status("Face model loading failed")
            self.camera.close()
            return
        if not self.camera.start_camera_stream():
            self.update_status("Failed to start camera stream")
            return
        self.update_status("Starting face recognition...")
        self.login_button.config(state=tk.DISABLED)
        self.is_running = True
        self.current_mode = "face_recognition"
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

    def processing_loop(self):
        try:
            while self.is_running:
                frame = self.camera.get_current_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                window_title = "System"
                if self.current_mode == "face_recognition":
                    self.process_face_recognition(frame)
                    window_title = "Face Recognition - Login"
                elif self.current_mode == "hand_tracking":
                    self.process_hand_tracking(frame)
                    window_title = f"Hand Tracking - {self.current_user}"
                
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    self.is_running = False
                elif key == ord('r'):
                    if self.current_mode == "face_recognition": self.register_new_face(frame)
                    elif self.current_mode == "hand_tracking": self.hand_manager.handle_key(key)
        except Exception as e:
            print(f"✗ Processing loop error: {e}")
        finally:
            self.is_running = False
            self.root.after(100, self.cleanup)

    def process_face_recognition(self, frame):
        _, face_result, recognition_result = self.face_manager.process_frame(frame)
        if face_result:
            x, y, w, h, _ = face_result
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if recognition_result and isinstance(recognition_result, tuple):
                best_match, _, login_ready, *optional_time = recognition_result
                if login_ready:
                    self.current_user = best_match
                    self.is_logged_in = True
                    self.current_mode = "hand_tracking"
                    self.root.after(0, self.on_login_success)
                elif optional_time:
                    cv2.putText(frame, f"Hold for {optional_time[0]:.1f}s", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif best_match:
                    cv2.putText(frame, f"Recognized: {best_match}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def process_hand_tracking(self, frame):
        self.hand_manager.process_frame(frame)
        cv2.putText(frame, f"User: {self.current_user}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def register_new_face(self, frame):
        if self.current_mode != "face_recognition": return
        face_result = self.face_manager.detect_face(frame)
        if not face_result: return
        x, y, w, h, _ = face_result
        face_roi = frame[y:y+h, x:x+w]
        embedding = self.face_manager.get_face_embedding(face_roi)
        if embedding is not None:
            name = self.simple_input_dialog("Enter name for the detected face:")
            if name:
                self.face_manager.register_face(name, embedding)
                self.face_manager.save_database()
                print(f"✓ Face registered for: {name}")

    def simple_input_dialog(self, prompt):
        dialog = tk.Toplevel(self.root)
        dialog.title("Register Face")
        dialog.transient(self.root); dialog.grab_set()
        result = [None]
        def on_ok():
            result[0] = entry.get()
            dialog.destroy()
        tk.Label(dialog, text=prompt).pack(pady=10)
        entry = tk.Entry(dialog, width=30); entry.pack(pady=5); entry.focus()
        btn_frame = tk.Frame(dialog); btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        entry.bind('<Return>', lambda e: on_ok())
        dialog.wait_window()
        return result[0]

    def on_login_success(self):
        self.update_status(f"Logged in as: {self.current_user}")
        self.user_info_label.config(text=f"Logged in as: {self.current_user}")
        self.logout_button.pack(side=tk.LEFT, padx=(0, 10))
        cv2.destroyAllWindows()
        self.update_status("Starting hand tracking...")
        if not self.hand_manager.initialize():
            self.update_status("Hand tracking initialization failed")
        else:
            self.update_status(f"Hand tracking active for: {self.current_user}")

    def logout(self):
        self.is_logged_in = False
        self.current_user = None
        self.current_mode = "idle"
        self.is_running = False 
        self.hand_manager.reset_mode_state()
        self.update_status("Logged out")
        self.user_info_label.config(text="Not logged in")
        self.logout_button.pack_forget()
        self.login_button.config(state=tk.NORMAL)
        cv2.destroyAllWindows()

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        print(f"Status: {message}")
    
    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("Cleaning up resources...")
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)
        self.camera.close()
        self.hand_manager.stop_screen_box_drawing()
        cv2.destroyAllWindows()
        print("✓ Application terminated")


def main():
    """Main entry point"""
    print("=== Integrated Face Recognition & Hand Tracking System ===")
    print("Starting GUI application...")
    
    # Set environment variables to prevent Qt timer issues
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
    
    app = IntegratedGUI()
    app.run()

if __name__ == "__main__":

    main() 
