import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import pygame
import threading

class KomonoDetector:
    def __init__(self):
        """Initialize the Komodo dragon detector."""
        print("Loading model...")
        self.model = YOLO('runs/detect/komodo_model/weights/best.pt')
        
        # Optimize model for inference
        self.model.fuse()  # Fuse model layers for faster inference
        
        # Initialize pygame for sound
        pygame.mixer.init()
        # Create alert sound
        self.create_alert_sound()
        
        print("Model loaded successfully!")
        print("Ready to detect Komodo dragons! Show a picture to the camera...")
    
    def create_alert_sound(self):
        """Create a notification sound file if it doesn't exist."""
        if not os.path.exists('alert.wav'):
            import wave
            import struct
            
            # Create a new WAV file
            sampleRate = 44100
            duration = 1  # seconds
            frequency = 440  # Hz
            
            wav_file = wave.open('alert.wav', 'w')
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)
            wav_file.setframerate(sampleRate)
            
            # Generate a more complex alert sound
            for i in range(int(duration * sampleRate)):
                # Create a more interesting sound with multiple frequencies
                value = 32767 * 0.3 * (
                    np.sin(2.0 * np.pi * frequency * i / sampleRate) +  # Base frequency
                    0.5 * np.sin(4.0 * np.pi * frequency * i / sampleRate) +  # First harmonic
                    0.25 * np.sin(6.0 * np.pi * frequency * i / sampleRate)   # Second harmonic
                )
                packed_value = struct.pack('h', int(value))
                wav_file.writeframes(packed_value)
            
            wav_file.close()
        
        # Load the sound file
        self.alert_sound = pygame.mixer.Sound('alert.wav')
    
    def play_alert(self):
        """Play alert sound when Komodo is detected."""
        try:
            # Play sound in a separate thread to avoid blocking
            threading.Thread(target=self._play_sound, daemon=True).start()
        except Exception as e:
            print(f"Error playing sound: {e}")
    
    def _play_sound(self):
        """Internal method to play the sound."""
        try:
            self.alert_sound.play()
            time.sleep(1)  # Let the sound play
        except Exception as e:
            print(f"Error in sound playback: {e}")
    
    def detect_from_camera(self):
        """Detect Komodo dragons from camera feed."""
        cap = cv2.VideoCapture(0)
        
        # Optimize camera settings for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Smaller resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS
        
        # Camera settings to reduce glare and improve visibility
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
        cap.set(cv2.CAP_PROP_CONTRAST, 60)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_GAIN, 40)
        
        print("\nStarting camera detection... Press 'q' to quit")
        print("Show a picture of a Komodo dragon to the camera!")
        print("Tips for better detection:")
        print("1. Hold the picture steady")
        print("2. Ensure even lighting - avoid direct light on the picture")
        print("3. Tilt the picture slightly if you see glare")
        print("4. Keep the picture about 1-2 feet from the camera")
        print("5. If the image is too dark, try moving closer to a light source")
        
        last_alert_time = 0  # Track when we last played the alert
        alert_cooldown = 2.0  # Seconds between alerts
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection with optimized settings
            results = self.model(frame, conf=0.35, iou=0.45, agnostic_nms=True)  # Lower base threshold
            
            # Process detections
            any_detections = False
            found_komodo = False
            highest_conf = 0
            
            for result in results:
                if len(result.boxes) > 0:
                    # Sort detections by confidence
                    confidences = [float(conf) for conf in result.boxes.conf]
                    sorted_indices = np.argsort(confidences)[::-1]
                    
                    for idx in sorted_indices[:3]:  # Only process top 3 detections for speed
                        confidence = float(result.boxes.conf[idx])
                        class_name = result.names[int(result.boxes.cls[idx])]
                        
                        # Show all detections above threshold
                        if confidence > 0.35:  # Lower minimum threshold
                            any_detections = True
                            
                            # Track highest confidence
                            if confidence > highest_conf:
                                highest_conf = confidence
                            
                            # Draw detection on frame
                            boxes = result.boxes.xyxy[idx].cpu().numpy()
                            
                            # Calculate box dimensions for shape analysis
                            box_width = boxes[2] - boxes[0]
                            box_height = boxes[3] - boxes[1]
                            aspect_ratio = box_width / box_height
                            area = box_width * box_height
                            frame_area = frame.shape[0] * frame.shape[1]
                            relative_size = area / frame_area
                            
                            # Human detection check first
                            is_likely_human = (
                                box_height > box_width or  # Vertical orientation
                                box_height > frame.shape[0] * 0.6 or  # Too tall
                                aspect_ratio < 1.2  # Human-like proportions
                            )
                            
                            # Balanced checks for Komodo dragon detection
                            if class_name.lower() == 'komodo dragon' and confidence > 0.45:  # Lower confidence threshold
                                if is_likely_human:
                                    color = (0, 0, 255)  # Red for rejected (human-like)
                                    thickness = 1
                                    print(f"Rejected: Likely human shape detected")
                                # Komodo dragons typically have:
                                # 1. More horizontal orientation (width > height)
                                # 2. Reasonable aspect ratio range
                                # 3. Minimum size requirement
                                elif (box_width > box_height * 0.8 and  # Allow slightly more vertical shapes
                                    1.3 < aspect_ratio < 2.3 and  # Wider aspect ratio range
                                    0.03 < relative_size < 0.8):  # More flexible size range
                                    found_komodo = True
                                    color = (0, 255, 0)  # Green for Komodo
                                    thickness = 3
                                    
                                    # Play alert sound with cooldown
                                    current_time = time.time()
                                    if current_time - last_alert_time > alert_cooldown:
                                        self.play_alert()
                                        last_alert_time = current_time
                                        print(f"Komodo dragon detected! Confidence: {confidence:.2f}")
                                else:
                                    color = (0, 255, 255)  # Yellow for uncertain
                                    thickness = 2
                                    reason = ""
                                    if not (box_width > box_height * 0.8):
                                        reason = "too vertical"
                                    elif not (1.3 < aspect_ratio < 2.3):
                                        reason = f"unusual shape (ratio: {aspect_ratio:.1f})"
                                    elif not (0.03 < relative_size < 0.8):
                                        reason = "unusual size"
                                    print(f"Possible Komodo but {reason}")
                            else:
                                color = (0, 0, 255)  # Red for other objects
                                thickness = 1
                            
                            # Draw box
                            cv2.rectangle(frame, 
                                        (int(boxes[0]), int(boxes[1])), 
                                        (int(boxes[2]), int(boxes[3])), 
                                        color, thickness)
                            
                            # Add text with better visibility
                            text = f"{class_name}: {confidence:.2f}"
                            if class_name.lower() == 'komodo dragon':
                                if is_likely_human:
                                    text += " (human-like shape)"
                                elif not found_komodo:
                                    if not (box_width > box_height * 0.8):
                                        text += " (too vertical)"
                                    elif not (1.3 < aspect_ratio < 2.3):
                                        text += f" (wrong proportions)"
                                    elif not (0.03 < relative_size < 0.8):
                                        text += " (wrong size)"
                            font_scale = 0.7 if found_komodo else 0.5
                            
                            # Add black background to text for better readability
                            (text_width, text_height), _ = cv2.getTextSize(text, 
                                                                         cv2.FONT_HERSHEY_SIMPLEX,
                                                                         font_scale, thickness)
                            text_x = int(boxes[0])
                            text_y = int(boxes[1]) - 5
                            cv2.rectangle(frame,
                                        (text_x, text_y - text_height),
                                        (text_x + text_width, text_y + 5),
                                        (0, 0, 0), -1)
                            
                            cv2.putText(frame, text,
                                      (text_x, text_y),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale,
                                      color, thickness)
            
            # Add instructions and status on the frame
            cv2.putText(frame,
                      "Show a Komodo dragon picture (Press 'q' to quit)",
                      (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, (255, 255, 255), 2)
            
            # Show appropriate message based on detection status
            if found_komodo:
                cv2.putText(frame,
                          f"Komodo dragon detected! ({highest_conf:.2f})",
                          (10, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, (0, 255, 0), 2)
            elif any_detections:
                cv2.putText(frame,
                          "No Komodo dragon detected - Try another angle",
                          (10, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Camera Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    detector = KomonoDetector()
    detector.detect_from_camera() 