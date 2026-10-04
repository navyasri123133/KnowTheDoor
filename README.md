\# KnowTheDoor



Know who is at the door, without turning visitors into a surveillance database.



A webcam program that recognizes people who agreed to be enrolled and shows their name,

arrival time, last visit, and a note. It alerts when an unknown person stays in view.



\## Install

1\. Install Python, then run: pip install -r requirements.txt

2\. Download two model files into recognition\\models:

&#x20;  - face\_detection\_yunet\_2023mar.onnx (Hugging Face: opencv/face\_detection\_yunet)

&#x20;  - face\_recognition\_sface\_2021dec.onnx (Hugging Face: opencv/face\_recognition\_sface)

3\. Run: python knowthedoor.py



\## Keys

\- e = enroll the person in view (asks for consent)

\- n = add a note

\- r = record a 20-second conversation (asks for consent, audio is not saved)

\- q = quit



\## Privacy

\- Only a face template is saved, never a photo.

\- Everything runs locally. Data is stored in recognition\\data and is not uploaded.

\- Alert snapshots blur every face.



\## Limits

\- A printed photo can fool the recognition.

\- Speech is English only, with no speaker separation.

\- The topic summary is keyword-based, not AI.

\- Data files are not encrypted.

\- This is a hackathon prototype, not a certified security product.

