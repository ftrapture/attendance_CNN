import os
import pickle
import gc
import logging
import time

import cv2
import numpy as np
import face_recognition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_liveness(image_frames, face_locations_list):
    if len(image_frames) < 3:
        return False, 0.0, "Insufficient frames for liveness detection"
    
    if len(face_locations_list) < 3:
        return False, 0.0, "Face not detected consistently"
    
    try:
        positions = []
        sizes = []
        for (top, right, bottom, left) in face_locations_list:
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            size = (right - left) * (bottom - top)
            positions.append([center_x, center_y])
            sizes.append(size)
        
        positions = np.array(positions)
        sizes = np.array(sizes)
        
        position_variance = np.var(positions, axis=0).sum()
        size_variance = np.var(sizes)
        
        if position_variance > 8000:
            return False, 0.0, "Excessive movement detected (possible video replay)"
        
        texture_scores = []
        for frame in image_frames:
            if frame is None or frame.size == 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            texture_scores.append(laplacian_var)
        
        if len(texture_scores) > 0:
            avg_texture = np.mean(texture_scores)
            if avg_texture < 10:
                return False, 0.0, "Low texture complexity (possible printed photo)"
        
        brightness_values = []
        for frame in image_frames:
            if frame is None or frame.size == 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            brightness_values.append(np.mean(gray))
        
        if len(brightness_values) >= 3:
            brightness_variance = np.var(brightness_values)
            pass
        
        if len(sizes) >= 3:
            size_std = np.std(sizes)
            size_mean = np.mean(sizes)
            size_cv = size_std / size_mean if size_mean > 0 else 0
            
            if size_cv > 0.35:
                return False, 0.0, "Inconsistent face size (possible screen display)"
        
        confidence = min(1.0, position_variance / 1000.0)
        return True, confidence, "Liveness verified"
        
    except Exception as e:
        logger.error(f"Liveness detection error: {e}")
        return False, 0.0, f"Liveness detection failed: {str(e)}"

def analyze_frame_sequence(frames):
    face_locations_list = []
    valid_frames = []
    
    for frame in frames:
        if frame is None or frame.size == 0:
            continue
        
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        brightness = np.mean(rgb_frame)
        if brightness < 100:
            rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.5, beta=30)
        
        face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=1)
        
        if len(face_locations) == 1:
            face_locations_list.append(face_locations[0])
            valid_frames.append(frame)
        elif len(face_locations) > 1:
            logger.warning("Multiple faces detected in frame")
        else:
            face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=2)
            if len(face_locations) == 1:
                face_locations_list.append(face_locations[0])
                valid_frames.append(frame)
            else:
                logger.warning("No face detected in frame")
    
    return valid_frames, face_locations_list

MODEL_PATH = "model.pkl"
_model_cache = None

def extract_face_encoding(image_path_or_array):
    try:
        if isinstance(image_path_or_array, str):
            image = face_recognition.load_image_file(image_path_or_array)
        else:
            if len(image_path_or_array.shape) == 3 and image_path_or_array.shape[2] == 3:
                image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
            else:
                image = image_path_or_array
        
        brightness = np.mean(image)
        if brightness < 100:
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
        elif brightness > 200:
            image = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(image, model="hog", number_of_times_to_upsample=1)
        
        if len(face_locations) == 0:
            face_locations = face_recognition.face_locations(image, model="hog", number_of_times_to_upsample=2)
        
        if len(face_locations) == 0:
            try:
                face_locations = face_recognition.face_locations(image, model="cnn")
            except:
                pass
        
        if len(face_locations) == 0:
            logger.warning("No face detected in image")
            return None
        
        encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(encodings) > 0:
            return encodings[0]
        else:
            logger.warning("No face encoding generated")
            return None
    except Exception as e:
        logger.error(f"Error extracting face encoding: {e}")
        return None

def extract_embedding_for_image(stream_or_bytes, require_liveness=False, additional_frames=None):
    data = None
    arr = None
    img = None
    try:
        data = stream_or_bytes.read()
        if not data:
            if require_liveness:
                return None, {"is_live": False, "reason": "No image data"}
            return None
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            if require_liveness:
                return None, {"is_live": False, "reason": "Invalid image"}
            return None
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        brightness = np.mean(rgb_img)
        if brightness < 100:
            rgb_img = cv2.convertScaleAbs(rgb_img, alpha=1.5, beta=30)
        elif brightness > 200:
            rgb_img = cv2.convertScaleAbs(rgb_img, alpha=0.8, beta=-20)
        
        face_locations = face_recognition.face_locations(rgb_img, model="hog", number_of_times_to_upsample=1)
        
        if len(face_locations) == 0:
            face_locations = face_recognition.face_locations(rgb_img, model="hog", number_of_times_to_upsample=2)
        
        if len(face_locations) == 0:
            try:
                face_locations = face_recognition.face_locations(rgb_img, model="cnn")
            except:
                pass
        
        if len(face_locations) == 0:
            if require_liveness:
                return None, {"is_live": False, "reason": "No face detected"}
            return None
        
        encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        if len(encodings) == 0:
            if require_liveness:
                return None, {"is_live": False, "reason": "No face encoding generated"}
            return None
        
        encoding = encodings[0]
        
        if encoding is None:
            if require_liveness:
                return None, {"is_live": False, "reason": "No face detected"}
            return None
        
        if require_liveness and additional_frames:
            all_frames = [img] + additional_frames
            valid_frames, face_locations = analyze_frame_sequence(all_frames)
            
            is_live, confidence, reason = detect_liveness(valid_frames, face_locations)
            
            liveness_result = {
                "is_live": is_live,
                "confidence": confidence,
                "reason": reason
            }
            
            if not is_live:
                return None, liveness_result
            
            return encoding, liveness_result
        
        if require_liveness:
            return encoding, {"is_live": True, "confidence": 1.0, "reason": "Single frame mode"}
        
        return encoding
    except Exception as e:
        logger.error(f"Error in extract_embedding_for_image: {e}")
        if require_liveness:
            return None, {"is_live": False, "reason": f"Error: {str(e)}"}
        return None
    finally:
        del data
        del arr
        del img
        gc.collect()

def load_model_if_exists():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    
    if not os.path.exists(MODEL_PATH):
        return None
    
    try:
        with open(MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
        logger.info("Model loaded successfully")
        return _model_cache
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        _model_cache = None
        return None

def predict_with_model(model_data, face_encoding, tolerance=0.6):
    try:
        if not model_data or 'encodings' not in model_data or 'labels' not in model_data:
            logger.error("Invalid model data")
            return None, 0.0
        
        known_encodings = model_data['encodings']
        known_labels = model_data['labels']
        
        if len(known_encodings) == 0:
            logger.warning("No known encodings in model")
            return None, 0.0
        
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        if min_distance <= tolerance:
            label = known_labels[min_distance_idx]
            confidence = 1.0 - min_distance
            return label, float(confidence)
        else:
            logger.warning(f"No match found. Minimum distance: {min_distance}")
            return None, 0.0
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise

def train_model_background(dataset_dir, progress_callback=None):
    encodings = []
    labels = []
    failed_images = 0
    processed_images = 0
    
    try:
        if progress_callback:
            progress_callback(5, "Scanning dataset", "scanning")
        
        if not os.path.exists(dataset_dir):
            logger.error(f"Dataset directory does not exist: {dataset_dir}")
            if progress_callback:
                progress_callback(0, "No training data found", "error")
            return
        
        student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        logger.info(f"Found student directories: {student_dirs}")
        
        if not student_dirs:
            logger.error("No student directories found in dataset")
            if progress_callback:
                progress_callback(0, "No training data found", "error")
            return
        
        total_students = len(student_dirs)
        processed = 0

        for sid in student_dirs:
            if progress_callback:
                pct = int((processed / total_students) * 60)
                progress_callback(pct, f"Processing student {processed + 1}/{total_students}", "loading")
            
            folder = os.path.join(dataset_dir, sid)
            logger.info(f"Processing folder: {folder}")
            try:
                files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                logger.info(f"Found {len(files)} image files in {folder}")
            except Exception as e:
                logger.error(f"Error reading folder {folder}: {e}")
                processed += 1
                continue
            
            if not files:
                logger.warning(f"No image files found in {folder}")
                processed += 1
                continue
            
            for fn in files:
                path = os.path.join(folder, fn)
                img = None
                try:
                    logger.info(f"Processing image: {path}")
                    img = cv2.imread(path)
                    if img is None:
                        logger.warning(f"Failed to read image: {path}")
                        failed_images += 1
                        continue
                    
                    encoding = extract_face_encoding(img)
                    
                    if encoding is not None:
                        processed_images += 1
                        encodings.append(encoding)
                        labels.append(sid)
                        logger.info(f"Successfully processed {fn} for student {sid}")
                    else:
                        logger.warning(f"No face detected in {fn}")
                        failed_images += 1
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    failed_images += 1
                    continue
                finally:
                    del img
                    gc.collect()
            
            processed += 1

        if len(encodings) < 1:
            logger.error("No valid face encodings found")
            if progress_callback:
                progress_callback(0, "No training data found", "error")
            return

        if progress_callback:
            progress_callback(70, f"Preparing data: {len(encodings)} images from {total_students} students", "preparing")

        model_data = {
            'encodings': encodings,
            'labels': labels
        }

        if progress_callback:
            progress_callback(90, "Saving model", "saving")

        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model_data, f)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            if progress_callback:
                progress_callback(0, "Error saving model", "error")
            return

        global _model_cache
        _model_cache = model_data

        del encodings
        del labels
        del model_data
        gc.collect()

        if progress_callback:
            progress_callback(100, f"Training complete! {processed_images} images from {total_students} students. Failed: {failed_images}", "complete")
        logger.info(f"Training complete. Images: {processed_images}, Students: {total_students}, Failed: {failed_images}")

    except Exception as e:
        logger.error(f"Critical error in training: {e}")
        if progress_callback:
            progress_callback(0, "Training error", "error")

