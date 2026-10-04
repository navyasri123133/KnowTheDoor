# KnowTheDoor

Know who is at the door, without turning visitors into a surveillance database.

KnowTheDoor is a webcam program for Windows. It recognizes people who agreed to be enrolled and shows
their name, arrival time, last visit, and a note. It raises an alert when an unknown person stays in view.

## Install

1. Install Python 3. In the project folder, install the packages:

```
pip install -r requirements.txt
```

2. Make a folder called models and download the two model files into it:

```
mkdir models
curl -L -o models\face_detection_yunet_2023mar.onnx https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx
curl -L -o models\face_recognition_sface_2021dec.onnx https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx
```

3. Start the program:

```
python knowthedoor.py
```

## Keys (click the video window first)

- e = enroll the one person in view (asks for consent)
- n = add a note for the recognized person
- r = record a 20-second conversation (asks for consent, audio is not saved)
- q = quit

## What it does

- Recognizes enrolled people and shows a card beside their face.
- Remembers arrival times and the last visit.
- Alerts with a red banner and a beep when an unknown face stays in view for 2 seconds.
- Saves alert snapshots with every face blurred, and logs events to data/events.csv.
- Turns a consented conversation into text on the laptop and shows a keyword summary.

## Privacy

- Only a face template (numbers) is saved, never a photo.
- Everything runs locally. The data folder is created on first run and is never uploaded.
- Enrollment and recording both ask for consent every time.
- To remove a person, delete their files in data/templates, data/notes, data/visits and data/talks.

## Limits

- A printed photo or a video of a person can fool the recognition.
- Accuracy drops in dim light, with masks, and with side views.
- Speech is English only, with no speaker separation.
- The topic summary is keyword-based, not AI.
- Saved data is not encrypted.
- This is a hackathon prototype, not a certified security product.
