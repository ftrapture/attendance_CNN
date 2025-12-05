import os
import pickle
import gc
import logging
import time

import cv2
import numpy as np
from deepface import DeepFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_liveness(image_frames, face_locations_list):
    if len(image_frames) < 2:
        return False, 0.0, "Insufficient frames for liveness detection"
    
    if len(face_locations_list) < 1:
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
        
        if position_variance > 20000:
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
            if avg_texture < 3:
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
        
        if len(sizes) >= 2:
            size_std = np.std(sizes)
            size_mean = np.mean(sizes)
            size_cv = size_std / size_mean if size_mean > 0 else 0
            
            if size_cv > 0.6:
                return False, 0.0, "Inconsistent face size (possible screen display)"
        
        confidence = min(1.0, position_variance / 1000.0)
        return True, confidence, "Liveness verified"
        
    except Exception as e:
        logger.error(f"Liveness detection error: {e}")
        return False, 0.0, f"Liveness detection failed: {str(e)}"

def analyze_frame_sequence(frames):
    face_locations_list = []
    valid_frames = []
    
    try:
        cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except:
        face_cascade = cv2.CascadeClassifier()
    
    for frame in frames:
        if frame is None or frame.size == 0:
            continue
        
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        brightness = np.mean(gray_frame)
        if brightness < 100:
            gray_frame = cv2.convertScaleAbs(gray_frame, alpha=1.5, beta=30)
        
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_locations_list.append((y, x+w, y+h, x))
            valid_frames.append(frame)
        elif len(faces) > 1:
            logger.warning("Multiple faces detected in frame")
        else:
            logger.warning("No face detected in frame")
    
    return valid_frames, face_locations_list

MODEL_PATH = "model.pkl"
_model_cache = None

def extract_face_encoding(image_path_or_array):
    try:
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
        else:
            img = image_path_or_array
        
        if img is None:
            logger.warning("Failed to load image")
            return None
        
        brightness = np.mean(img)
        if brightness < 100:
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
        elif brightness > 200:
            img = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
        
        try:
            result = DeepFace.represent(
                img_path=img,
                model_name='ArcFace',
                enforce_detection=False,
                detector_backend='retinaface',
                align=True
            )
            
            if result and len(result) > 0:
                embedding = result[0]['embedding']
                return np.array(embedding)
            else:
                logger.warning("No face detected in image")
                return None
                
        except Exception as e:
            logger.warning(f"RetinaFace detection failed: {e}, trying opencv")
            try:
                result = DeepFace.represent(
                    img_path=img,
                    model_name='ArcFace',
                    enforce_detection=False,
                    detector_backend='opencv',
                    align=True
                )
                if result and len(result) > 0:
                    return np.array(result[0]['embedding'])
            except:
                pass
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
        
        bgr_img = img
        
        brightness = np.mean(bgr_img)
        if brightness < 100:
            bgr_img = cv2.convertScaleAbs(bgr_img, alpha=1.5, beta=30)
        elif brightness > 200:
            bgr_img = cv2.convertScaleAbs(bgr_img, alpha=0.8, beta=-20)
        
        try:
            result = DeepFace.represent(
                img_path=bgr_img,
                model_name='ArcFace',
                enforce_detection=False,
                detector_backend='retinaface',
                align=True
            )
            
            if result and len(result) > 0:
                encoding = np.array(result[0]['embedding'])
            else:
                if require_liveness:
                    return None, {"is_live": False, "reason": "No face detected"}
                return None
                
        except Exception as e:
            logger.warning(f"RetinaFace detection failed: {e}, trying opencv")
            try:
                result = DeepFace.represent(
                    img_path=bgr_img,
                    model_name='ArcFace',
                    enforce_detection=False,
                    detector_backend='opencv',
                    align=True
                )
                if result and len(result) > 0:
                    encoding = np.array(result[0]['embedding'])
                else:
                    if require_liveness:
                        return None, {"is_live": False, "reason": "No face detected"}
                    return None
            except:
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

def predict_with_model(model_data, face_encoding, tolerance=0.65):
    try:
        if not model_data or 'encodings' not in model_data or 'labels' not in model_data:
            logger.error("Invalid model data")
            return None, 0.0
        
        known_encodings = model_data['encodings']
        known_labels = model_data['labels']
        
        if len(known_encodings) == 0:
            logger.warning("No known encodings in model")
            return None, 0.0
        
        if face_encoding is None or len(face_encoding) == 0:
            logger.error("Invalid face encoding provided")
            return None, 0.0
        
        if np.any(np.isnan(face_encoding)) or np.any(np.isinf(face_encoding)):
            logger.error("Face encoding contains NaN or Inf values")
            return None, 0.0
        
        from scipy.spatial import distance
        
        best_match_idx = -1
        best_similarity = -1
        second_best_similarity = -1
        
        for idx, known_encoding in enumerate(known_encodings):
            try:
                if known_encoding is None or len(known_encoding) == 0:
                    continue
                
                if len(known_encoding) != len(face_encoding):
                    logger.warning(f"Encoding size mismatch: {len(known_encoding)} vs {len(face_encoding)}")
                    continue
                
                cosine_dist = distance.cosine(face_encoding, known_encoding)
                
                if np.isnan(cosine_dist) or np.isinf(cosine_dist):
                    logger.warning(f"Invalid distance for encoding {idx}")
                    continue
                
                similarity = 1 - cosine_dist
                
                if similarity > best_similarity:
                    second_best_similarity = best_similarity
                    best_similarity = similarity
                    best_match_idx = idx
                elif similarity > second_best_similarity:
                    second_best_similarity = similarity
            except Exception as comp_error:
                logger.warning(f"Error comparing with encoding {idx}: {comp_error}")
                continue
        
        if best_similarity >= tolerance:
            margin = best_similarity - second_best_similarity
            if second_best_similarity > 0 and margin < 0.08:
                logger.warning(f"Insufficient margin between top 2 matches: {margin:.3f}. Best: {best_similarity:.3f}, Second: {second_best_similarity:.3f}")
                return None, float(best_similarity)
            
            label = known_labels[best_match_idx]
            return label, float(best_similarity)
        else:
            logger.warning(f"No match found. Best similarity: {best_similarity}")
            return None, float(best_similarity) if best_similarity >= 0 else 0.0
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return None, 0.0

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
                        logger.warning(f"No face detected in {fn} - skipping")
                        failed_images += 1
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    failed_images += 1
                finally:
                    if img is not None:
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

        logger.info(f"Total encodings created: {len(encodings)}")
        logger.info(f"Total unique students: {len(set(labels))}")
        
        for sid in set(labels):
            count = labels.count(sid)
            logger.info(f"Student {sid}: {count} images processed")

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

        total_encodings = len(encodings)
        
        del encodings
        del labels
        del model_data
        gc.collect()

        if progress_callback:
            progress_callback(100, f"Training complete! {processed_images} images from {total_students} students (Failed: {failed_images})", "complete")
        logger.info(f"Training complete. Images: {processed_images}, Students: {total_students}, Failed: {failed_images}")
        logger.info(f"Model contains {total_encodings} encodings")

    except Exception as e:
        logger.error(f"Critical error in training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if progress_callback:
            progress_callback(0, f"Training error: {str(e)}", "error")
    finally:
        try:
            if 'encodings' in locals():
                del encodings
            if 'labels' in locals():
                del labels
            if 'model_data' in locals():
                del model_data
            gc.collect()
        except:
            pass

