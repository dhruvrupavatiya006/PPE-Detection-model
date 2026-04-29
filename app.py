from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import pandas as pd
import sqlite3

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = YOLO("runs/detect/train2/weights/best.pt")
classes = model.names


# ---------------- DATABASE SETUP ----------------
def init_db():
    conn = sqlite3.connect("safety.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS safety_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT,
            helmet INTEGER,
            gloves INTEGER,
            vest INTEGER,
            status TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


init_db()

# ---------------- GLOBAL ANALYSIS DATA ----------------
analysis_data = []


def is_inside(item_box, person_box):
    px1, py1, px2, py2 = person_box
    x1, y1, x2, y2 = item_box[:4]
    return px1 < x1 < px2 and py1 < y1 < py2


def generate_frames():
    global analysis_data
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, conf=0.4, persist=True)[0]

        persons = []
        ppe_positive = {"helmet": [], "gloves": [], "vest": []}
        ppe_negative = {"no_helmet": [], "no_gloves": []}

        for box in results.boxes:
            cls = classes[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if cls == "Person":
                persons.append((x1, y1, x2, y2))
            elif cls in ppe_positive:
                ppe_positive[cls].append((x1, y1, x2, y2, conf))
            elif cls in ppe_negative:
                ppe_negative[cls].append((x1, y1, x2, y2, conf))

        analysis_data = []

        # -------- OPEN DB CONNECTION (PER FRAME) --------
        conn = sqlite3.connect("safety.db")
        c = conn.cursor()

        analysis_data = []

        conn = sqlite3.connect("safety.db")
        c = conn.cursor()

        for box in results.boxes:
            cls = classes[int(box.cls[0])]
            if cls != "Person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            worker_id = f"Person_{track_id}"
            status = {"helmet": False, "gloves": False, "vest": False}
            confs = []

            for item in ppe_positive:
                for pbox in ppe_positive[item]:
                    if is_inside(pbox, (x1, y1, x2, y2)):
                        status[item] = True
                        confs.append(pbox[4])
            for neg in ppe_negative:
                for nbox in ppe_negative[neg]:
                    if is_inside(nbox, (x1, y1, x2, y2)):
                        if neg == "no_helmet":
                            status["helmet"] = False
                        elif neg == "no_gloves":
                            status["gloves"] = False
            avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
            safe = all(status.values())

            analysis_data.append(
                {
                    "Person": worker_id,
                    "Helmet": status["helmet"],
                    "Gloves": status["gloves"],
                    "Vest": status["vest"],
                    "Status": "Safe" if safe else "Unsafe",
                    "Confidence": avg_conf,
                }
            )

            c.execute(
                """
                INSERT INTO safety_logs
                (worker_id, helmet, gloves, vest, status, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    worker_id,
                    int(status["helmet"]),
                    int(status["gloves"]),
                    int(status["vest"]),
                    "Safe" if safe else "Unsafe",
                    avg_conf,
                ),
            )  # Commit the database transaction
            color = (0, 255, 0) if safe else (0, 0, 255)
            label = f"{worker_id} | {'SAFE' if safe else 'UNSAFE'} ({avg_conf})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
        conn.commit()
        conn.close()

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/demo")
def demo():
    return render_template("monitor.html")


@app.route("/dashboard_data")
def dashboard_data():
    df = pd.DataFrame(analysis_data)
    stats = {
        "total": len(df),
        "safe": (df["Status"] == "Safe").sum() if not df.empty else 0,
        "unsafe": (df["Status"] == "Unsafe").sum() if not df.empty else 0,
    }
    return render_template(
        "dashboard_table.html",
        table=df.to_dict(orient="records"),
        stats=stats,
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/technology")
def technology():
    return render_template("technology.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
