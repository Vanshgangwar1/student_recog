import json
import os
import sys
import time
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from urllib import error, request

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODEL_DIR = DATA_DIR / "model"
STUDENTS_FILE = DATA_DIR / "students.json"
TRAINER_FILE = MODEL_DIR / "trainer.yml"
TIMETABLE_FILE = DATA_DIR / "timetable.json"
ALERTS_DIR = DATA_DIR / "alerts"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_IMAGE_SIZE = (100, 100)
DETECTION_RESIZE_SCALE = 0.6
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_WARMUP_FRAMES = 5
CAPTURE_SAMPLE_INTERVAL_SECONDS = 0.18
CAPTURE_DUPLICATE_THRESHOLD = 10.0
RECOGNITION_FRAME_SKIP = 1
RECOGNITION_STABLE_FRAMES = 3
IGNORED_ACTIVITIES = {
    "break",
    "lunch",
    "sports",
    "sport",
    "yoga",
    "net lab",
    "library",
    "club activity",
}
ALERT_COOLDOWN_SECONDS = 600
JSON_CACHE = {}


def default_timetable() -> dict:
    weekdays = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    template = [
        {"start": "09:00", "end": "09:50", "activity": "Math"},
        {"start": "09:50", "end": "10:40", "activity": "Science"},
        {"start": "10:40", "end": "11:00", "activity": "Break"},
        {"start": "11:00", "end": "11:50", "activity": "English"},
        {"start": "11:50", "end": "12:40", "activity": "Social Studies"},
        {"start": "12:40", "end": "13:20", "activity": "Lunch"},
        {"start": "13:20", "end": "14:10", "activity": "Computer"},
        {"start": "14:10", "end": "15:00", "activity": "Library"},
    ]
    return {
        "default_section": "A",
        "timezone_note": "Times are interpreted using the computer's local clock.",
        "sections": {
            "A": {
                "days": {
                    weekday: ([*template] if weekday != "sunday" else [])
                    for weekday in weekdays
                }
            }
        },
    }


def read_json_file(file_path: Path, default):
    if not file_path.exists():
        return default

    cache_key = str(file_path)
    try:
        mtime_ns = file_path.stat().st_mtime_ns
    except OSError:
        return default

    cached_entry = JSON_CACHE.get(cache_key)
    if cached_entry and cached_entry["mtime_ns"] == mtime_ns:
        return cached_entry["data"]

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        JSON_CACHE[cache_key] = {"mtime_ns": mtime_ns, "data": data}
        return data
    except json.JSONDecodeError:
        return default


def write_json_file(file_path: Path, data: dict) -> None:
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    JSON_CACHE[str(file_path)] = {
        "mtime_ns": file_path.stat().st_mtime_ns,
        "data": data,
    }


def ensure_directories() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    if not STUDENTS_FILE.exists():
        STUDENTS_FILE.write_text("{}", encoding="utf-8")
    if not TIMETABLE_FILE.exists():
        write_json_file(TIMETABLE_FILE, default_timetable())

def require_lbph():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "LBPH recognizer is unavailable.\n"
            "Install it with: pip install opencv-contrib-python"
        )
    return cv2.face.LBPHFaceRecognizer_create()


def load_students() -> dict:
    ensure_directories()
    return read_json_file(STUDENTS_FILE, {})


def save_students(students: dict) -> None:
    write_json_file(STUDENTS_FILE, students)


def load_timetable() -> dict:
    ensure_directories()
    return read_json_file(TIMETABLE_FILE, default_timetable())


def get_sections() -> dict:
    timetable = load_timetable()
    return timetable.get("sections", {})


def get_default_section() -> str:
    timetable = load_timetable()
    default_section = timetable.get("default_section", "").strip()
    sections = get_sections()
    if default_section and default_section in sections:
        return default_section
    if sections:
        return next(iter(sections))
    return ""


def resolve_section(section_name: str | None = None) -> str:
    sections = get_sections()
    if not sections:
        return ""

    if section_name:
        normalized = section_name.strip().upper()
        for existing in sections:
            if existing.upper() == normalized:
                return existing

    return get_default_section()


def get_face_detector():
    detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    return detector


def prepare_face_image(face_region) -> np.ndarray:
    face_region = cv2.resize(face_region, FACE_IMAGE_SIZE)
    return cv2.equalizeHist(face_region)


def detect_faces(detector, gray_frame):
    small_frame = cv2.resize(
        gray_frame,
        None,
        fx=DETECTION_RESIZE_SCALE,
        fy=DETECTION_RESIZE_SCALE,
        interpolation=cv2.INTER_LINEAR,
    )
    small_frame = cv2.equalizeHist(small_frame)
    detected_faces = detector.detectMultiScale(
        small_frame,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(40, 40),
    )

    scale_back = 1.0 / DETECTION_RESIZE_SCALE
    faces = []
    for (x, y, w, h) in detected_faces:
        faces.append(
            (
                int(x * scale_back),
                int(y * scale_back),
                int(w * scale_back),
                int(h * scale_back),
            )
        )
    return faces


def configure_camera(camera) -> None:
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_camera(index: int = 0):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    configure_camera(camera)
    for _ in range(CAMERA_WARMUP_FRAMES):
        camera.read()
    return camera


def face_difference_score(first_face: np.ndarray | None, second_face: np.ndarray) -> float:
    if first_face is None:
        return float("inf")
    return float(np.mean(cv2.absdiff(first_face, second_face)))


def compute_training_signature(label_map: dict) -> str:
    hasher = sha256()

    for label, student_id in sorted(label_map.items()):
        hasher.update(f"{label}:{student_id}|".encode("utf-8"))
        student_dir = DATASET_DIR / student_id
        for image_path in sorted(student_dir.glob("*.jpg")):
            stat = image_path.stat()
            hasher.update(
                f"{image_path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode("utf-8")
            )

    return hasher.hexdigest()


def save_training_metadata(metadata: dict) -> None:
    write_json_file(MODEL_DIR / "training_meta.json", metadata)


def load_training_metadata() -> dict:
    return read_json_file(MODEL_DIR / "training_meta.json", {})


def should_skip_training(label_map: dict) -> bool:
    if not TRAINER_FILE.exists():
        return False

    metadata = load_training_metadata()
    current_signature = compute_training_signature(label_map)
    return metadata.get("signature") == current_signature


def capture_student_faces(
    student_id: str,
    name: str,
    guardian_phone: str,
    section: str = "",
    samples: int = 10,
) -> None:
    detector = get_face_detector()
    students = load_students()
    existing_student = students.get(student_id, {})
    students[student_id] = {
        "name": name,
        "guardian_phone": guardian_phone,
        "section": section or existing_student.get("section", ""),
    }
    save_students(students)

    student_dir = DATASET_DIR / student_id
    student_dir.mkdir(parents=True, exist_ok=True)

    camera = open_camera()

    print(f"\nCapturing face samples for {name} ({student_id})")
    print("Press 'q' to stop early.\n")

    saved = 0
    last_saved_at = 0.0
    last_saved_face = None

    while saved < samples:
        success, frame = camera.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)

        for (x, y, w, h) in faces:
            face_region = gray[y : y + h, x : x + w]
            face_region = prepare_face_image(face_region)
            now = time.time()
            difference_score = face_difference_score(last_saved_face, face_region)

            if now - last_saved_at < CAPTURE_SAMPLE_INTERVAL_SECONDS:
                continue
            if difference_score < CAPTURE_DUPLICATE_THRESHOLD:
                continue

            file_path = student_dir / f"{saved + 1:03d}.jpg"
            cv2.imwrite(str(file_path), face_region)
            saved += 1
            last_saved_at = now
            last_saved_face = face_region

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                frame,
                f"Samples: {saved}/{samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (50, 255, 50),
                2,
            )
            cv2.putText(
                frame,
                f"Change: {difference_score:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 220),
                2,
            )
            break

        cv2.imshow("Capture Student Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} face image(s) to {student_dir}")


def load_training_data():
    students = load_students()
    faces = []
    labels = []
    label_map = {}
    next_label = 1

    for student_id in sorted(students.keys()):
        student_dir = DATASET_DIR / student_id
        if not student_dir.exists():
            continue

        image_paths = sorted(student_dir.glob("*.jpg"))
        if not image_paths:
            continue

        label_map[next_label] = student_id
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            faces.append(prepare_face_image(image))
            labels.append(next_label)

        next_label += 1

    return faces, labels, label_map


def save_label_map(label_map: dict) -> None:
    label_map_path = MODEL_DIR / "labels.json"
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")


def load_label_map() -> dict:
    label_map_path = MODEL_DIR / "labels.json"
    if not label_map_path.exists():
        return {}
    return read_json_file(label_map_path, {})


def parse_minutes(value: str) -> int:
    hour_text, minute_text = value.split(":")
    return int(hour_text) * 60 + int(minute_text)


def get_day_slots(section_name: str | None = None, day_name: str | None = None) -> list:
    schedule = load_timetable()
    resolved_section = resolve_section(section_name)
    day_name = day_name or datetime.now().strftime("%A").lower()

    if resolved_section:
        section_days = schedule.get("sections", {}).get(resolved_section, {}).get("days", {})
        if day_name in section_days:
            return section_days.get(day_name, [])

    return schedule.get("days", {}).get(day_name, schedule.get(day_name, []))


def get_current_schedule_slot(
    now: datetime | None = None, section_name: str | None = None
) -> dict | None:
    now = now or datetime.now()
    day_name = now.strftime("%A").lower()
    day_slots = get_day_slots(section_name, day_name)
    current_minutes = now.hour * 60 + now.minute

    for slot in day_slots:
        try:
            start = parse_minutes(slot["start"])
            end = parse_minutes(slot["end"])
        except (KeyError, ValueError):
            continue
        if start <= current_minutes < end:
            return slot
    return None


def is_ignored_activity(activity: str) -> bool:
    normalized = activity.strip().lower()
    return any(ignored in normalized for ignored in IGNORED_ACTIVITIES)


def get_monitoring_decision(
    now: datetime | None = None, section_name: str | None = None
) -> tuple[bool, str, dict | None]:
    resolved_section = resolve_section(section_name)
    slot = get_current_schedule_slot(now, resolved_section)
    if not slot:
        if resolved_section:
            return (
                False,
                f"No active timetable slot right now for section {resolved_section}.",
                None,
            )
        return False, "No timetable slot is active right now.", None

    activity = slot.get("activity", "Unknown")
    if is_ignored_activity(activity):
        if resolved_section:
            return False, f"{resolved_section}: monitoring paused for {activity}.", slot
        return False, f"Monitoring paused for {activity}.", slot

    if resolved_section:
        return True, f"{resolved_section}: monitoring during {activity}.", slot
    return True, f"Monitoring during {activity}.", slot


def save_alert_snapshot(frame, student_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alert_path = ALERTS_DIR / f"{student_id}_{timestamp}.jpg"
    cv2.imwrite(str(alert_path), frame)
    return alert_path


def send_alert(student_id: str, student: dict, slot: dict | None, image_path: Path) -> bool:
    webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return False

    activity = slot.get("activity", "class period") if slot else "class period"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (
        f"Possible bunking alert: {student.get('name', 'Unknown Student')} "
        f"(ID: {student_id}) was detected during {activity} at {timestamp}."
    )
    payload = {
        "student_id": student_id,
        "student_name": student.get("name", ""),
        "section": student.get("section", ""),
        "guardian_phone": student.get("guardian_phone", ""),
        "activity": activity,
        "timestamp": timestamp,
        "message": message,
        "image_path": str(image_path),
        "image_url": "",
    }

    image_base_url = os.getenv("ALERT_IMAGE_BASE_URL", "").rstrip("/")
    if image_base_url:
        payload["image_url"] = f"{image_base_url}/{image_path.name}"

    req = request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    token = os.getenv("ALERT_WEBHOOK_TOKEN", "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with request.urlopen(req, timeout=10) as response:
            return 200 <= response.status < 300
    except (error.URLError, error.HTTPError):
        return False


def train_model() -> None:
    recognizer = require_lbph()
    faces, labels, label_map = load_training_data()

    if not faces:
        raise RuntimeError(
            "No training data found. Register a student and capture face samples first."
        )

    if should_skip_training(label_map):
        print("Training skipped because the dataset has not changed.")
        print(f"Existing trained model is already up to date at {TRAINER_FILE}")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(str(TRAINER_FILE))
    save_label_map(label_map)
    save_training_metadata(
        {
            "signature": compute_training_signature(label_map),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "image_count": len(faces),
            "student_count": len(label_map),
        }
    )

    print(f"Model trained successfully with {len(faces)} images.")
    print(f"Saved trained model to {TRAINER_FILE}")


def recognize_faces(section_name: str, confidence_threshold: float = 70.0) -> None:
    if not TRAINER_FILE.exists():
        raise RuntimeError("Trained model not found. Please train the model first.")

    resolved_section = resolve_section(section_name)
    if not resolved_section:
        raise RuntimeError("No timetable sections found. Please check timetable.json.")

    recognizer = require_lbph()
    recognizer.read(str(TRAINER_FILE))
    label_map = load_label_map()
    students = load_students()
    detector = get_face_detector()

    camera = open_camera()

    print(f"\nRunning live recognition for section {resolved_section}.")
    print("Press 'q' to quit.\n")

    last_alert_times = {}
    frame_index = 0
    recognized_faces = []
    last_status_refresh_minute = None
    monitoring_enabled = False
    status_text = ""
    active_slot = None
    recognition_history = {}

    while True:
        success, frame = camera.read()
        if not success:
            continue

        frame_index += 1
        current_minute = datetime.now().strftime("%Y%m%d%H%M")
        if current_minute != last_status_refresh_minute:
            monitoring_enabled, status_text, active_slot = get_monitoring_decision(
                section_name=resolved_section
            )
            last_status_refresh_minute = current_minute

        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255) if monitoring_enabled else (200, 200, 0),
            2,
        )

        if not monitoring_enabled:
            cv2.imshow("Student Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        if frame_index % (RECOGNITION_FRAME_SKIP + 1) == 1 or not recognized_faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(detector, gray)
            new_recognized_faces = []
            next_recognition_history = {}

            for (x, y, w, h) in faces:
                face_region = gray[y : y + h, x : x + w]
                face_region = prepare_face_image(face_region)
                label, confidence = recognizer.predict(face_region)

                student_id = label_map.get(str(label)) or label_map.get(label)
                student = students.get(student_id, {})
                student_section = resolve_section(student.get("section"))

                identity_key = student_id if confidence < confidence_threshold and student else "unknown"
                history_count = recognition_history.get(identity_key, 0) + 1
                next_recognition_history[identity_key] = history_count
                is_stable = history_count >= RECOGNITION_STABLE_FRAMES

                if confidence < confidence_threshold and student and is_stable:
                    if student_section and student_section != resolved_section:
                        text = (
                            f"{student['name']} | ID: {student_id} | "
                            f"Section: {student_section} (watching {resolved_section})"
                        )
                        color = (0, 165, 255)
                    else:
                        text = (
                            f"{student['name']} | ID: {student_id} | "
                            f"Section: {student_section or resolved_section}"
                        )
                        color = (50, 220, 50)
                    if not student_section or student_section == resolved_section:
                        current_time = datetime.now().timestamp()
                        last_alert_time = last_alert_times.get(student_id, 0.0)
                        if current_time - last_alert_time >= ALERT_COOLDOWN_SECONDS:
                            image_path = save_alert_snapshot(frame, student_id)
                            alert_sent = send_alert(student_id, student, active_slot, image_path)
                            last_alert_times[student_id] = current_time
                            print(
                                f"Alert {'sent' if alert_sent else 'saved locally'} for "
                                f"{student['name']} ({student_id}) at {image_path}"
                            )
                else:
                    text = "Scanning..." if not is_stable else "Unknown"
                    color = (0, 215, 255) if not is_stable else (20, 20, 255)

                new_recognized_faces.append(
                    {
                        "box": (x, y, w, h),
                        "text": text,
                        "color": color,
                        "confidence": confidence,
                    }
                )

            recognized_faces = new_recognized_faces
            recognition_history = next_recognition_history

        for recognized_face in recognized_faces:
            x, y, w, h = recognized_face["box"]
            color = recognized_face["color"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                recognized_face["text"],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {recognized_face['confidence']:.1f}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("Student Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def prompt_student_details():
    student_id = input("Enter student ID: ").strip()
    name = input("Enter student name: ").strip()
    guardian_phone = input("Enter guardian phone number: ").strip()
    available_sections = ", ".join(get_sections().keys())
    section = input(
        f"Enter student section ({available_sections or 'section not configured'}): "
    ).strip()

    if not student_id or not name:
        raise ValueError("Student ID and student name are required.")

    resolved_section = resolve_section(section)
    if get_sections() and not resolved_section:
        raise ValueError("Student section is required and must match timetable.json.")

    return student_id, name, guardian_phone, resolved_section


def prompt_monitor_section() -> str:
    sections = get_sections()
    if not sections:
        return ""

    default_section = get_default_section()
    options = ", ".join(sections.keys())
    value = input(
        f"Enter section to monitor [{default_section}] ({options}): "
    ).strip()
    resolved = resolve_section(value or default_section)
    if not resolved:
        raise ValueError("Invalid section selected.")
    return resolved


def print_menu() -> None:
    print("\nStudent Recognition System")
    print("1. Register student and capture face samples")
    print("2. Train recognition model")
    print("3. Start timetable-aware live monitoring")
    print("4. Show timetable and monitoring status")
    print("5. Exit")


def show_monitoring_status() -> None:
    section = prompt_monitor_section()
    monitor, status_text, slot = get_monitoring_decision(section_name=section)
    print(f"\nTimetable file: {TIMETABLE_FILE}")
    print(f"Selected section: {section or 'default'}")
    print(status_text)
    if slot:
        print(
            "Active slot: "
            f"{slot.get('start', '--:--')} - {slot.get('end', '--:--')} | "
            f"{slot.get('activity', 'Unknown')}"
        )
    else:
        print("No active slot right now.")
    print(f"Monitoring enabled: {'Yes' if monitor else 'No'}")
    print("Available sections:", ", ".join(get_sections().keys()))
    print("Ignored activities:", ", ".join(sorted(IGNORED_ACTIVITIES)))
    print("To enable SMS alerts, set ALERT_WEBHOOK_URL in your environment.")
    print(
        "To include a public image link in the alert payload, set ALERT_IMAGE_BASE_URL "
        "to a URL where alert images are hosted."
    )


def main() -> None:
    ensure_directories()

    while True:
        print_menu()
        choice = input("Choose an option: ").strip()

        try:
            if choice == "1":
                student_id, name, guardian_phone, section = prompt_student_details()
                capture_student_faces(student_id, name, guardian_phone, section)
            elif choice == "2":
                train_model()
            elif choice == "3":
                section = prompt_monitor_section()
                recognize_faces(section)
            elif choice == "4":
                show_monitoring_status()
            elif choice == "5":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please select 1, 2, 3, 4 or 5.")
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        sys.exit(0)
