import json
import os
import sys
from datetime import datetime
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
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json_file(file_path: Path, data: dict) -> None:
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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


def capture_student_faces(
    student_id: str,
    name: str,
    guardian_phone: str,
    section: str = "",
    samples: int = 25,
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

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print(f"\nCapturing face samples for {name} ({student_id})")
    print("Press 'q' to stop early.\n")

    saved = 0

    while saved < samples:
        success, frame = camera.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_region = gray[y : y + h, x : x + w]
            face_region = cv2.resize(face_region, (200, 200))

            file_path = student_dir / f"{saved + 1:03d}.jpg"
            cv2.imwrite(str(file_path), face_region)
            saved += 1

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

    for label, student_id in enumerate(sorted(students.keys()), start=1):
        student_dir = DATASET_DIR / student_id
        if not student_dir.exists():
            continue

        label_map[label] = student_id
        for image_path in student_dir.glob("*.jpg"):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            faces.append(image)
            labels.append(label)

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

    recognizer.train(faces, np.array(labels))
    recognizer.save(str(TRAINER_FILE))
    save_label_map(label_map)

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

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print(f"\nRunning live recognition for section {resolved_section}.")
    print("Press 'q' to quit.\n")

    last_alert_times = {}

    while True:
        success, frame = camera.read()
        if not success:
            continue

        monitoring_enabled, status_text, active_slot = get_monitoring_decision(
            section_name=resolved_section
        )

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_region = gray[y : y + h, x : x + w]
            face_region = cv2.resize(face_region, (200, 200))
            label, confidence = recognizer.predict(face_region)

            student_id = label_map.get(str(label)) or label_map.get(label)
            student = students.get(student_id, {})
            student_section = resolve_section(student.get("section"))

            if confidence < confidence_threshold and student:
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
                text = "Unknown"
                color = (20, 20, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}",
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
