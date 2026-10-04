import csv
import glob
import json
import os
import re
import sys
import textwrap
import time
from datetime import date, datetime

import cv2
import numpy as np

try:
    import winsound            # Windows beep for the alert
except ImportError:
    winsound = None

try:                           # optional: only needed for the speech feature
    import sounddevice as sd
    from faster_whisper import WhisperModel
    SPEECH_OK = True
except Exception:
    SPEECH_OK = False

# ---- Settings ----
DET_MODEL = "models/face_detection_yunet_2023mar.onnx"
REC_MODEL = "models/face_recognition_sface_2021dec.onnx"
TEMPLATE_DIR = "data/templates"
NOTES_DIR = "data/notes"
VISITS_DIR = "data/visits"
TALKS_DIR = "data/talks"
ALERT_DIR = "data/alerts"
EVENT_LOG = "data/events.csv"
MATCH_THRESHOLD = 0.363   # higher = more similar; tune on your team
PROCESS_EVERY = 3         # recognise on every 3rd frame to stay fast
VISIT_GAP = 60            # seconds away before coming back counts as a NEW visit
BLUE = (255, 128, 0)      # OpenCV uses BGR, so this is a normal blue
TYPE_SPEED = 60           # characters per second for the line-by-line reveal
RESTART_AFTER = 2         # seconds away before the reveal starts again
UNKNOWN_SECONDS = 2       # an unknown face must stay this long to raise an alert
ALERT_COOLDOWN = 30       # seconds between alerts
BANNER_SECONDS = 5        # how long the red alert banner stays
RECORD_SECONDS = 20       # length of one recorded conversation
WHISPER_MODEL = "tiny.en" # small English speech model (about 75 MB, runs on CPU)
# ------------------

COSINE = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)

for path in (DET_MODEL, REC_MODEL):
    if not os.path.exists(path):
        sys.exit(f"Model file missing: {path}\nPut the two .onnx files in the 'models' folder.")

os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)
os.makedirs(VISITS_DIR, exist_ok=True)
os.makedirs(TALKS_DIR, exist_ok=True)
os.makedirs(ALERT_DIR, exist_ok=True)

detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320), 0.85, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(REC_MODEL, "")


def load_templates():
    templates = {}
    for file in glob.glob(os.path.join(TEMPLATE_DIR, "*.npy")):
        name = os.path.splitext(os.path.basename(file))[0]
        templates[name] = np.load(file)
    return templates


def analyse(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    results = []
    if faces is not None:
        for face in faces:
            aligned = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(aligned)
            results.append((face, feature))
    return results


def identify(feature, templates):
    best_name, best_score = None, -1.0
    for name, template in templates.items():
        score = recognizer.match(feature, template, COSINE)
        if score > best_score:
            best_name, best_score = name, score
    if best_score >= MATCH_THRESHOLD:
        return best_name, best_score
    return None, best_score


def enroll(feature):
    name = input("Name of the person (letters and numbers only): ").strip()
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        print("Enrollment cancelled: use letters, numbers, or underscore only.")
        return False
    consent = input(f"Does {name} agree to be enrolled? Type 'yes' to confirm: ").strip().lower()
    if consent != "yes":
        print("Enrollment cancelled: no consent.")
        return False
    np.save(os.path.join(TEMPLATE_DIR, f"{name}.npy"), feature)
    print(f"Enrolled {name}. Only a template was saved, not a photo.")
    return True


# ---- Messages shown after a person is recognised ----
def get_note(name):
    path = os.path.join(NOTES_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_note(name, text):
    path = os.path.join(NOTES_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": str(date.today()), "note": text}, f)


def load_visits(name):
    path = os.path.join(VISITS_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def record_visit(name, when):
    """Save this arrival time. Return the PREVIOUS visit time (or None)."""
    visits = load_visits(name)
    previous = visits[-1] if visits else None
    visits.append(when.isoformat(timespec="seconds"))
    with open(os.path.join(VISITS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(visits[-50:], f)
    return previous


def log_event(event, detail=""):
    new_file = not os.path.exists(EVENT_LOG)
    with open(EVENT_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "event", "detail"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), event, detail])


def save_alert_snapshot(raw, faces):
    """Save the frame with EVERY face blurred, so the alert keeps no identifiable faces."""
    img = raw.copy()
    H, W = img.shape[:2]
    for face in faces:
        x, y, w, h = [int(v) for v in face[:4]]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (51, 51), 30)
    path = os.path.join(ALERT_DIR, datetime.now().strftime("unknown_%Y%m%d_%H%M%S.jpg"))
    cv2.imwrite(path, img)
    return path


def draw_banner(frame, text, now):
    W = frame.shape[1]
    colour = (0, 0, 220) if int(now * 2) % 2 == 0 else (0, 0, 120)
    cv2.rectangle(frame, (0, 0), (W, 34), colour, -1)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


STOPWORDS = set("about after also because been being could from going have here into just know like "
                "more really said should some than that their them then there they think this very were "
                "what when where which while will with would your".split())


def load_talks(name):
    path = os.path.join(TALKS_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_talk(name, text):
    talks = load_talks(name)
    talks.append({"time": datetime.now().isoformat(timespec="seconds"), "text": text})
    with open(os.path.join(TALKS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(talks[-20:], f)


def summarize(talks):
    """Simple keyword summary of past talks. This is NOT an AI language model."""
    if not talks:
        return []
    counts = {}
    for word in re.findall(r"[a-z']+", " ".join(t["text"].lower() for t in talks)):
        if len(word) > 3 and word not in STOPWORDS:
            counts[word] = counts.get(word, 0) + 1
    top = sorted(counts, key=lambda w: (-counts[w], w))[:5]
    last = talks[-1]
    when = datetime.fromisoformat(last["time"]).strftime("%d %b")
    snippet = last["text"].strip()
    if len(snippet) > 70:
        snippet = snippet[:67] + "..."
    lines = []
    if top:
        lines.append("Topics: " + ", ".join(top))
    lines.append(f"Last talk ({when}): {snippet}")
    return lines


_stt = None


def transcribe(audio):
    global _stt
    if _stt is None:
        print("Loading speech model (the first time it downloads about 75 MB)...")
        _stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = _stt.transcribe(audio, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def start_recording(name, now):
    audio = sd.rec(int(RECORD_SECONDS * 16000), samplerate=16000, channels=1, dtype="float32")
    return {"name": name, "start": now, "audio": audio}


def finish_recording(rec):
    sd.wait()
    audio = rec["audio"][:, 0]
    try:
        text = transcribe(audio)
    except Exception as error:
        print("Speech to text failed:", error)
        return
    if not text:
        print("No speech heard. Nothing saved.")
        return
    save_talk(rec["name"], text)
    log_event("talk_saved", f"{rec['name']}: {len(text)} characters")
    print(f"Saved conversation for {rec['name']}: {text}")


def message_for(name, info=None):
    lines = [f"Name: {name}"]
    if info:
        lines.append("Arrived: " + info["arrived"].strftime("%d %b, %I:%M:%S %p"))
        prev = info["previous"]
        if prev:
            lines.append("Last visit: " + datetime.fromisoformat(prev).strftime("%d %b, %I:%M %p"))
        else:
            lines.append("Last visit: first time")
    lines += summarize(load_talks(name))
    note = get_note(name)
    lines.append(("Note: " + note["note"]) if note else "No note yet. Press n to add one.")
    return lines


def draw_card(frame, x, y, w, h, lines, elapsed, side_memory, key):
    """Draw the data card beside the face (left or right, never on it).
    Lines appear one after another, typed out over time."""
    H, W = frame.shape[:2]
    gap, min_w, max_w, line_h = 10, 150, 300, 20
    space_right = W - (x + w) - gap
    space_left = x - gap

    # pick a side: keep the old side while it still has room, else use the roomier side
    side = side_memory.get(key)
    if side == "right" and space_right >= min_w:
        pass
    elif side == "left" and space_left >= min_w:
        pass
    elif space_right >= min_w or space_left >= min_w:
        side = "right" if space_right >= space_left else "left"
    else:
        side = "above"
    side_memory[key] = side

    if side == "right":
        card_w = min(max_w, space_right)
        cx = x + w + gap
    elif side == "left":
        card_w = min(max_w, space_left)
        cx = x - gap - card_w
    else:
        card_w = min(max_w, W - 20)
        cx = max(0, min(x, W - card_w))

    chars = max(12, int((card_w - 16) / 9.5))
    wrapped = []
    for line in lines:
        wrapped += textwrap.wrap(line, chars) or [""]
    wrapped = wrapped[:10]
    full_h = 14 + line_h * len(wrapped)

    if side == "above":
        cy = y - gap - full_h
        if cy < 0:
            cy = min(H - full_h, y + h + gap)
    else:
        cy = max(0, min(y, H - full_h))

    # typing effect: reveal characters over time, one line after another
    remaining = int(elapsed * TYPE_SPEED) + 1
    visible = []
    for line in wrapped:
        if remaining <= 0:
            break
        visible.append(line[:remaining])
        remaining -= len(line)

    card_h = 14 + line_h * len(visible)
    overlay = frame.copy()
    cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), BLUE, 1)
    for i, line in enumerate(visible):
        colour = (255, 200, 120) if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (cx + 8, cy + 6 + line_h * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return (cx, cy, card_w, full_h)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit("Camera not found. Close Zoom/Teams/browser tabs using it, or try index 1.")

    templates = load_templates()
    print(f"Loaded {len(templates)} enrolled people. e = enroll, n = note, r = record talk, q = quit.")
    last = []
    last_seen = {}
    appeared_at = {}
    visit_info = {}
    side_memory = {}
    msg_cache = {}
    unknown_since = None
    last_alert = -1e9
    banner_until = 0
    recording = None
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read a frame from the camera.")
            break

        if frame_count % PROCESS_EVERY == 0:
            last = analyse(frame)
        frame_count += 1

        raw = frame.copy()
        known_names = []
        unknown_faces = []
        now = time.time()
        for face, feature in last:
            x, y, w, h = [int(v) for v in face[:4]]
            name, score = identify(feature, templates)
            if name:
                known_names.append(name)
                colour, label = BLUE, ""
                away = now - last_seen.get(name, 0)
                if away > RESTART_AFTER:
                    appeared_at[name] = now
                if away > VISIT_GAP:
                    when = datetime.now()
                    previous = record_visit(name, when)
                    visit_info[name] = {"arrived": when, "previous": previous}
                    log_event("visit", name)
                    print(" | ".join(message_for(name, visit_info[name])))
                last_seen[name] = now
            else:
                unknown_faces.append(face)
                colour, label = (0, 0, 255), f"UNKNOWN ({score:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 1)
            if label:
                cv2.putText(frame, label, (x, max(y - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        for face, feature in last:
            name, _ = identify(feature, templates)
            if name:
                x, y, w, h = [int(v) for v in face[:4]]
                elapsed = now - appeared_at.get(name, now)
                cached = msg_cache.get(name)
                if cached is None or now - cached[0] > 1:
                    cached = (now, message_for(name, visit_info.get(name)))
                    msg_cache[name] = cached
                draw_card(frame, x, y, w, h, cached[1], elapsed, side_memory, name)

        # security alert: an unknown face that stays in view
        if unknown_faces:
            if unknown_since is None:
                unknown_since = now
            elif now - unknown_since >= UNKNOWN_SECONDS and now - last_alert > ALERT_COOLDOWN:
                path = save_alert_snapshot(raw, [f for f, _ in last])
                log_event("unknown_alert", path)
                print(f"ALERT: unknown person. Blurred snapshot saved: {path}")
                last_alert = now
                banner_until = now + BANNER_SECONDS
                if winsound:
                    try:
                        winsound.Beep(1000, 300)
                    except Exception:
                        pass
        else:
            unknown_since = None
        if now < banner_until:
            draw_banner(frame, "ALERT: UNKNOWN PERSON", now)

        # recording indicator and finishing a recording
        if recording:
            left = RECORD_SECONDS - (now - recording["start"])
            cv2.circle(frame, (20, frame.shape[0] - 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {max(0, int(left))}s  {recording['name']}",
                        (36, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if left <= 0:
                cv2.putText(frame, "Transcribing...", (36, frame.shape[0] - 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("KnowTheDoor (e enroll, n note, r record, q quit)", frame)
                cv2.waitKey(1)
                finish_recording(recording)
                recording = None

        cv2.imshow("KnowTheDoor (e enroll, n note, r record, q quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("e"):
            if len(last) == 1:
                if enroll(last[0][1]):
                    templates = load_templates()
            else:
                print(f"Need exactly one face in view to enroll (seeing {len(last)}).")
        if key == ord("n"):
            if len(known_names) == 1:
                text = input(f"Message for {known_names[0]} (English letters only): ").strip()
                clean = text.encode("ascii", "ignore").decode()
                if clean != text:
                    print("Some characters can't be shown on the video and were removed.")
                if clean:
                    save_note(known_names[0], clean)
                    log_event("note_saved", known_names[0])
                    print(f"Saved. It will show next time {known_names[0]} is recognised.")
            else:
                print("Need exactly one recognised person in view to add a note.")
        if key == ord("r"):
            if not SPEECH_OK:
                print("Speech packages are not installed. Run: pip install faster-whisper sounddevice")
            elif recording:
                print("Already recording.")
            elif len(known_names) == 1:
                who = known_names[0]
                agreed = input(f"Does {who} agree to have this conversation turned into text? "
                               "Audio is NOT saved. Type 'yes': ").strip().lower()
                if agreed == "yes":
                    recording = start_recording(who, time.time())
                    print(f"Recording {RECORD_SECONDS} seconds. Speak now...")
                else:
                    print("Cancelled: no consent.")
            else:
                print("Need exactly one recognised person in view to record.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()