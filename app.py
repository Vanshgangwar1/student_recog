import json
import sys
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODEL_DIR = DATA_DIR / "model"
STUDENTS_FILE = DATA_DIR / "students.json"
TRAINER_FILE = MODEL_DIR / "trainer.yml"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def ensure_directories() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not STUDENTS_FILE.exists():
        STUDENTS_FILE.write_text("{}", encoding="utf-8")


def require_lbph():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "LBPH recognizer is unavailable.\n"
            "Install it with: pip install opencv-contrib-python"
        )
    return cv2.face.LBPHFaceRecognizer_create()


def load_students() -> dict:
    ensure_directories()
    try:
        return json.loads(STUDENTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_students(students: dict) -> None:
    STUDENTS_FILE.write_text(json.dumps(students, indent=2), encoding="utf-8")


def get_face_detector():
    detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    return detector


def capture_student_faces(student_id: str, name: str, guardian_phone: str, samples: int = 25) -> None:
    detector = get_face_detector()
    students = load_students()
    students[student_id] = {
        "name": name,
        "guardian_phone": guardian_phone,
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
    return json.loads(label_map_path.read_text(encoding="utf-8"))


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


def recognize_faces(confidence_threshold: float = 70.0) -> None:
    if not TRAINER_FILE.exists():
        raise RuntimeError("Trained model not found. Please train the model first.")

    recognizer = require_lbph()
    recognizer.read(str(TRAINER_FILE))
    label_map = load_label_map()
    students = load_students()
    detector = get_face_detector()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("\nRunning live recognition.")
    print("Press 'q' to quit.\n")

    while True:
        success, frame = camera.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_region = gray[y : y + h, x : x + w]
            face_region = cv2.resize(face_region, (200, 200))
            label, confidence = recognizer.predict(face_region)

            student_id = label_map.get(str(label)) or label_map.get(label)
            student = students.get(student_id, {})

            if confidence < confidence_threshold and student:
                text = f"{student['name']} | ID: {student_id}"
                color = (50, 220, 50)
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

    if not student_id or not name:
        raise ValueError("Student ID and student name are required.")

    return student_id, name, guardian_phone


def print_menu() -> None:
    print("\nStudent Recognition System")
    print("1. Register student and capture face samples")
    print("2. Train recognition model")
    print("3. Start live face recognition")
    print("4. Exit")


def main() -> None:
    ensure_directories()

    while True:
        print_menu()
        choice = input("Choose an option: ").strip()

        try:
            if choice == "1":
                student_id, name, guardian_phone = prompt_student_details()
                capture_student_faces(student_id, name, guardian_phone)
            elif choice == "2":
                train_model()
            elif choice == "3":
                recognize_faces()
            elif choice == "4":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please select 1, 2, 3 or 4.")
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        sys.exit(0)
