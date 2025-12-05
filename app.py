import os
import io
import threading
import sqlite3
import datetime
import json
import time
import pytz
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from model import train_model_background, extract_embedding_for_image, MODEL_PATH, load_model_if_exists, predict_with_model, extract_face_encoding

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")
LOCAL_TZ = pytz.timezone('Asia/Kolkata')

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Please compress images before uploading."}), 413

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        roll TEXT,
                        class TEXT,
                        section TEXT,
                        reg_no TEXT,
                        has_faces INTEGER DEFAULT 0,
                        created_at TEXT
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        check_in_time TEXT NOT NULL,
                        check_out_time TEXT,
                        is_late INTEGER DEFAULT 0,
                        duration_minutes INTEGER DEFAULT 0,
                        confidence REAL DEFAULT 0.0,
                        FOREIGN KEY(student_id) REFERENCES students(id)
                    )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(check_in_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id)")
        
        conn.commit()
    except Exception as e:
        app.logger.error(f"DB init error: {e}")
    finally:
        conn.close()

@app.before_request
def ensure_db():
    init_db()

def write_train_status(status_dict):
    try:
        with open(TRAIN_STATUS_FILE, "w") as f:
            json.dump(status_dict, f)
    except Exception as e:
        app.logger.error(f"Failed to write train status: {e}")

def read_train_status():
    try:
        if not os.path.exists(TRAIN_STATUS_FILE):
            return {"running": False, "progress": 0, "message": "Not trained", "stage": "idle"}
        with open(TRAIN_STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        app.logger.error(f"Failed to read train status: {e}")
        return {"running": False, "progress": 0, "message": "Error reading status", "stage": "error"}

def calculate_late_threshold():
    return datetime.time(9, 30)

write_train_status({"running": False, "progress": 0, "message": "No training yet.", "stage": "idle"})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    conn = get_db()
    try:
        df = pd.read_sql_query("SELECT check_in_time FROM attendance", conn)
        if df.empty:
            days = [(datetime.datetime.now(LOCAL_TZ).date() - datetime.timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
            return jsonify({"dates": days, "counts": [0]*30})
        df['date'] = pd.to_datetime(df['check_in_time'], format='ISO8601').dt.date
        last_30 = [(datetime.datetime.now(LOCAL_TZ).date() - datetime.timedelta(days=i)) for i in range(29, -1, -1)]
        counts = [int(df[df['date'] == d].shape[0]) for d in last_30]
        dates = [d.strftime("%d-%b") for d in last_30]
        return jsonify({"dates": dates, "counts": counts})
    except Exception as e:
        app.logger.error(f"attendance_stats error: {e}")
        days = [(datetime.datetime.now(LOCAL_TZ).date() - datetime.timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
        return jsonify({"dates": days, "counts": [0]*30})
    finally:
        conn.close()

@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")
    data = request.form
    name = data.get("name","").strip()
    roll = data.get("roll","").strip()
    cls = data.get("class","").strip()
    sec = data.get("sec","").strip()
    reg_no = data.get("reg_no","").strip()
    
    if not name:
        return jsonify({"error":"Full name is required"}), 400
    if not roll:
        return jsonify({"error":"Roll number is required"}), 400
    if not cls:
        return jsonify({"error":"Class is required"}), 400
    if not sec:
        return jsonify({"error":"Section is required"}), 400
    if not reg_no:
        return jsonify({"error":"Registration number is required"}), 400
    
    name_parts = name.split()
    if len(name_parts) < 2:
        return jsonify({"error":"Please enter full name including surname (e.g., John Doe)"}), 400
    
    conn = get_db()
    try:
        c = conn.cursor()
        
        c.execute("SELECT id FROM students WHERE name = ?", (name,))
        if c.fetchone():
            return jsonify({"error": f"Student '{name}' already exists"}), 409
        
        c.execute("SELECT id, name FROM students WHERE roll = ?", (roll,))
        existing = c.fetchone()
        if existing:
            return jsonify({"error": f"Roll number '{roll}' is already assigned to {existing[1]}"}), 409
        
        c.execute("SELECT id, name FROM students WHERE reg_no = ?", (reg_no,))
        existing = c.fetchone()
        if existing:
            return jsonify({"error": f"Registration number '{reg_no}' is already assigned to {existing[1]}"}), 409
            
    except Exception as e:
        app.logger.error("add_student error: %s", e)
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
    
    import json
    temp_data = {
        "name": name,
        "roll": roll,
        "class": cls,
        "section": sec,
        "reg_no": reg_no
    }
    return jsonify({"student_id": "temp_" + str(int(time.time() * 1000)), "temp_data": temp_data, "message": "Student info validated. Please upload images."})

@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error":"student_id required"}), 400
    
    temp_data_json = request.form.get("temp_data")
    temp_data = {}
    if temp_data_json:
        try:
            temp_data = json.loads(temp_data_json)
        except:
            pass
    
    files = request.files.getlist("images[]")
    app.logger.info("Received %d files", len(files))
    if not files:
        return jsonify({"error":"no images"}), 400
    
    file_data = []
    all_embeddings = []
    for file in files:
        if not file or file.filename == "":
            app.logger.warning("Skipping empty file")
            continue
        try:
            app.logger.info("Processing file: %s", file.filename)
            file_bytes = file.read()
            app.logger.info("File size: %d bytes", len(file_bytes))
            
            img = Image.open(io.BytesIO(file_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            img_array = np.array(img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            import face_recognition
            rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img, model="hog")
            
            if len(face_locations) == 0:
                app.logger.warning("No face detected in: %s", file.filename)
                return jsonify({
                    "error": f"No face detected in {file.filename}. Please upload a clear photo with a visible face."
                }), 400
            elif len(face_locations) > 1:
                app.logger.warning("Multiple faces detected in: %s", file.filename)
                return jsonify({
                    "error": f"Multiple faces detected in {file.filename}. Please upload photos with only ONE person."
                }), 400
            
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 5:
                return jsonify({
                    "error": f"{file.filename} appears to be AI-generated or animated. Please use real photos only."
                }), 400
            
            hist = cv2.calcHist([img_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            unique_colors = np.count_nonzero(hist > 0.001)
            
            if unique_colors < 25:
                return jsonify({
                    "error": f"{file.filename} appears to be animated or cartoon. Please use real photos only."
                }), 400
            
            encoding = extract_face_encoding(img_bgr)
            
            if encoding is not None:
                app.logger.info("Face detected and encoded for: %s", file.filename)
                all_embeddings.append(encoding)
                file_data.append((file.filename, file_bytes))
            else:
                app.logger.warning("No face detected in: %s", file.filename)
                return jsonify({
                    "error": f"No face detected in {file.filename}. Please upload a clear photo with a visible face."
                }), 400
        except Exception as e:
            app.logger.error("Error processing image: %s", e)
            import traceback
            app.logger.error(traceback.format_exc())
            continue
    
    app.logger.info("Total files processed: %d, embeddings: %d, file_data: %d", len(files), len(all_embeddings), len(file_data))
    
    if len(all_embeddings) == 0:
        embeddings_arr = np.array([])
    else:
        embeddings_arr = np.array(all_embeddings)
    
    if len(embeddings_arr) > 1:
        import face_recognition
        distances = []
        for i in range(1, len(embeddings_arr)):
            dist = face_recognition.face_distance([embeddings_arr[0]], embeddings_arr[i])[0]
            distances.append(dist)
        avg_dist = np.mean(distances)
        if avg_dist > 0.65:
            return jsonify({"error": "Images appear to be different people. Upload images of the same person"}), 400
    
    app.logger.info("Checking for duplicate faces across all students...")
    import face_recognition
    if os.path.exists(DATASET_DIR) and os.listdir(DATASET_DIR):
        for existing_sid in os.listdir(DATASET_DIR):
            existing_folder = os.path.join(DATASET_DIR, existing_sid)
            if not os.path.isdir(existing_folder):
                continue
            if existing_sid == student_id:
                continue
            
            for fname in os.listdir(existing_folder):
                try:
                    fpath = os.path.join(existing_folder, fname)
                    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    
                    existing_encoding = extract_face_encoding(img)
                    if existing_encoding is None:
                        continue
                    
                    for new_emb in embeddings_arr:
                        dist = face_recognition.face_distance([existing_encoding], new_emb)[0]
                        if dist < 0.45:
                            return jsonify({"error": f"This image matches with an existing student ID {existing_sid}. Please upload different pictures"}), 400
                except Exception as e:
                    app.logger.error("Error checking existing image: %s", e)
                    continue
    
    is_temp = student_id.startswith("temp_")
    
    temp_folder = os.path.join(DATASET_DIR, student_id)
    os.makedirs(temp_folder, exist_ok=True)
    
    folder = temp_folder
    os.makedirs(folder, exist_ok=True)
    
    existing_embeddings = []
    import face_recognition
    if os.path.exists(folder) and len(os.listdir(folder)) > 0:
        for fname in os.listdir(folder):
            try:
                fpath = os.path.join(folder, fname)
                img = cv2.imread(fpath, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                existing_encoding = extract_face_encoding(img)
                if existing_encoding is not None:
                    existing_embeddings.append(existing_encoding)
            except Exception as e:
                app.logger.error("Error loading existing image: %s", e)
                continue
    
    if existing_embeddings:
        for new_emb in embeddings_arr:
            distances = face_recognition.face_distance(existing_embeddings, new_emb)
            min_dist = np.min(distances)
            if min_dist < 0.45:
                return jsonify({"error": "This image already exists in this student's records. Please upload different pictures"}), 400
    
    saved = 0
    app.logger.info("Saving %d files to folder %s", len(file_data), folder)
    for filename, file_bytes in file_data:
        try:
            fname = f"{datetime.datetime.utcnow().timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            app.logger.info("Saving file to: %s", path)
            with open(path, 'wb') as f:
                f.write(file_bytes)
            app.logger.info("File saved successfully: %s", path)
            saved += 1
        except Exception as e:
            app.logger.error("save error: %s", e)
            import traceback
            app.logger.error(traceback.format_exc())
    
    app.logger.info("Total files saved: %d", saved)
    
    if saved == 0:
        if os.path.exists(folder):
            import shutil
            shutil.rmtree(folder, ignore_errors=True)
        return jsonify({"error": "No images were saved"}), 400
    
    if is_temp and temp_data:
        conn = get_db()
        try:
            c = conn.cursor()
            now = datetime.datetime.now(LOCAL_TZ).isoformat()
            c.execute("INSERT INTO students (name, roll, class, section, reg_no, has_faces, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
                      (temp_data.get("name"), temp_data.get("roll"), temp_data.get("class"), temp_data.get("section"), temp_data.get("reg_no"), now))
            actual_student_id = c.lastrowid
            conn.commit()
            
            new_folder = os.path.join(DATASET_DIR, str(actual_student_id))
            if os.path.exists(folder) and folder != new_folder:
                import shutil
                shutil.move(folder, new_folder)
            
            app.logger.info("Student saved to DB with ID: %d", actual_student_id)
        except sqlite3.IntegrityError:
            if os.path.exists(folder):
                import shutil
                shutil.rmtree(folder, ignore_errors=True)
            return jsonify({"error": f"Student '{temp_data.get('name')}' already exists"}), 409
        except Exception as e:
            app.logger.error("Database save error: %s", e)
            if os.path.exists(folder):
                import shutil
                shutil.rmtree(folder, ignore_errors=True)
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()
    else:
        actual_student_id = int(student_id)
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute("UPDATE students SET has_faces = 1 WHERE id = ?", (actual_student_id,))
            conn.commit()
        finally:
            conn.close()
    
    return jsonify({"saved": saved, "message": f"Successfully registered student with {saved} images"})

@app.route("/train_model", methods=["GET"])
def train_model_route():
    try:
        status = read_train_status()
        if status.get("running"):
            return jsonify({"status":"already_running", "progress": status.get("progress", 0)}), 202
        write_train_status({"running": True, "progress": 0, "message": "Initializing", "stage": "init"})
        t = threading.Thread(
            target=train_model_background, 
            args=(DATASET_DIR, lambda p,m,s: write_train_status({"running": True, "progress": p, "message": m, "stage": s})),
            daemon=True
        )
        t.start()
        return jsonify({"status":"started"}), 202
    except Exception as e:
        app.logger.error(f"train_model error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    return render_template("mark_attendance.html")

@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    try:
        if "image" not in request.files:
            return jsonify({"recognized": False, "error":"no image"}), 400
        img_file = request.files["image"]
        if not img_file or img_file.filename == '':
            return jsonify({"recognized": False, "error":"invalid image file"}), 400
        
        additional_frames = []
        frame_count = 0
        
        while f"frame{frame_count}" in request.files:
            frame_file = request.files[f"frame{frame_count}"]
            if frame_file and frame_file.filename != '':
                frame_bytes = frame_file.read()
                frame_arr = np.frombuffer(frame_bytes, np.uint8)
                frame_img = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
                if frame_img is not None:
                    additional_frames.append(frame_img)
            frame_count += 1
        
        require_liveness = len(additional_frames) >= 2
        
        if require_liveness:
            result = extract_embedding_for_image(img_file.stream, require_liveness=True, additional_frames=additional_frames)
            if result is None or result[0] is None:
                liveness_info = result[1] if result else {"reason": "Processing failed"}
                return jsonify({
                    "recognized": False, 
                    "error": "Liveness check failed",
                    "liveness": liveness_info
                }), 200
            
            emb, liveness_result = result
            
            if not liveness_result.get("is_live", False):
                return jsonify({
                    "recognized": False,
                    "error": "Please be physically present for attendance",
                    "liveness": liveness_result
                }), 200
        else:
            app.logger.warning("Single frame mode - liveness detection disabled")
            emb = extract_embedding_for_image(img_file.stream)
            if emb is None:
                return jsonify({"recognized": False, "error":"no face detected"}), 200
            liveness_result = {"is_live": False, "reason": "Single frame mode"}
        
        clf = load_model_if_exists()
        if clf is None:
            return jsonify({"recognized": False, "error":"model not trained"}), 200
        
        try:
            pred_label, conf = predict_with_model(clf, emb)
            app.logger.info(f"Prediction result: label={pred_label}, confidence={conf}")
        except Exception as e:
            app.logger.error(f"prediction error: {e}")
            return jsonify({"recognized": False, "error":"prediction failed"}), 500
        
        if pred_label is None:
            app.logger.warning(f"No match found. Best confidence: {conf}")
            return jsonify({"recognized": False, "confidence": float(conf), "error": "No matching student found"}), 200
        
        if conf < 0.35:
            app.logger.warning(f"Confidence too low: {conf} for student {pred_label}")
            return jsonify({"recognized": False, "confidence": float(conf), "student_id": int(pred_label), "error": "Confidence too low - please try again"}), 200
        
        student_id = int(pred_label)
        
        conn = get_db()
        try:
            c = conn.cursor()
            local_now = datetime.datetime.now(LOCAL_TZ)
            today_str = local_now.strftime("%Y-%m-%d")
            
            c.execute("""SELECT id, check_out_time FROM attendance 
                         WHERE student_id = ? AND check_in_time LIKE ?""",
                      (student_id, f"{today_str}%"))
            row = c.fetchone()
            
            if row:
                if row[1] is None:
                    check_out_time = local_now.isoformat()
                    c.execute("""UPDATE attendance SET check_out_time = ? WHERE id = ?""",
                              (check_out_time, row[0]))
                    conn.commit()
                    c.execute("SELECT name FROM students WHERE id=?", (student_id,))
                    name_row = c.fetchone()
                    name = name_row[0] if name_row else "Unknown"
                    return jsonify({
                        "recognized": True, 
                        "student_id": student_id, 
                        "name": name,
                        "confidence": float(conf),
                        "status": "check_out",
                        "liveness": liveness_result,
                        "message": "Check-out successful"
                    }), 200
                else:
                    c.execute("SELECT name FROM students WHERE id=?", (student_id,))
                    name_row = c.fetchone()
                    name = name_row[0] if name_row else "Unknown"
                    return jsonify({
                        "recognized": True,
                        "student_id": student_id,
                        "name": name,
                        "confidence": float(conf),
                        "status": "already_marked",
                        "liveness": liveness_result,
                        "message": "Already given attendance today"
                    }), 200
            
            c.execute("SELECT name FROM students WHERE id=?", (student_id,))
            name_row = c.fetchone()
            name = name_row[0] if name_row else "Unknown"
            
            check_in_time = local_now.isoformat()
            is_late = 1 if local_now.time() > calculate_late_threshold() else 0
            
            c.execute("""INSERT INTO attendance 
                         (student_id, name, check_in_time, is_late, confidence) 
                         VALUES (?, ?, ?, ?, ?)""",
                      (student_id, name, check_in_time, is_late, float(conf)))
            conn.commit()
            
            return jsonify({
                "recognized": True,
                "student_id": student_id,
                "name": name,
                "confidence": float(conf),
                "status": "check_in",
                "is_late": is_late,
                "liveness": liveness_result,
                "message": "Attendance marked successfully"
            }), 200
        except Exception as e:
            app.logger.error(f"attendance error: {e}")
            return jsonify({"recognized": False, "error": "failed to process attendance"}), 500
        finally:
            conn.close()
    except Exception as e:
        app.logger.exception("recognize_face error")
        return jsonify({"recognized": False, "error": str(e)}), 500

@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    try:
        period = request.args.get("period", "all")
        conn = get_db()
        try:
            c = conn.cursor()
            q = "SELECT id, student_id, name, check_in_time, check_out_time, is_late, duration_minutes, confidence FROM attendance"
            params = ()
            if period == "daily":
                today_str = datetime.datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
                q += " WHERE check_in_time LIKE ?"
                params = (f"{today_str}%",)
            elif period == "weekly":
                start_date = (datetime.datetime.now(LOCAL_TZ) - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                q += " WHERE check_in_time >= ?"
                params = (f"{start_date}",)
            elif period == "monthly":
                start_date = (datetime.datetime.now(LOCAL_TZ) - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
                q += " WHERE check_in_time >= ?"
                params = (f"{start_date}",)
            q += " ORDER BY check_in_time DESC LIMIT 5000"
            c.execute(q, params)
            rows = c.fetchall()
            return render_template("attendance_record.html", records=rows, period=period)
        finally:
            conn.close()
    except Exception as e:
        app.logger.error(f"attendance_record error: {e}")
        return render_template("attendance_record.html", records=[], period="all")

@app.route("/download_csv", methods=["GET"])
def download_csv():
    try:
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute("SELECT id, student_id, name, check_in_time, check_out_time, is_late, duration_minutes, confidence FROM attendance ORDER BY check_in_time DESC")
            rows = c.fetchall()
            output = io.StringIO()
            output.write("id,student_id,name,check_in_time,check_out_time,is_late,duration_minutes,confidence\n")
            for r in rows:
                output.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4] or ""},{r[5]},{r[6] or 0},{r[7] or 0}\n')
            mem = io.BytesIO()
            mem.write(output.getvalue().encode("utf-8"))
            mem.seek(0)
            return send_file(mem, as_attachment=True, download_name="attendance.csv", mimetype="text/csv")
        finally:
            conn.close()
    except Exception as e:
        app.logger.error(f"download_csv error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/students", methods=["GET"])
def students_list():
    try:
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute("SELECT id, name, roll, class, section, reg_no, created_at FROM students ORDER BY id DESC")
            rows = c.fetchall()
            data = [{"id":r[0],"name":r[1],"roll":r[2],"class":r[3],"section":r[4],"reg_no":r[5],"created_at":r[6]} for r in rows]
            return jsonify({"students": data})
        finally:
            conn.close()
    except Exception as e:
        app.logger.error(f"students_list error: {e}")
        return jsonify({"students": [], "error": str(e)}), 500

@app.route("/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    try:
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute("DELETE FROM students WHERE id=?", (sid,))
            c.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
            conn.commit()
            folder = os.path.join(DATASET_DIR, str(sid))
            if os.path.isdir(folder):
                import shutil
                shutil.rmtree(folder, ignore_errors=True)
            return jsonify({"deleted": True})
        except Exception as e:
            app.logger.error(f"delete_student error: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()
    except Exception as e:
        app.logger.error(f"delete_student connection error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)