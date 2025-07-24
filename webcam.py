# python3 -m venv venv
# source venv/bin/activate
# python3 -m pip install opencv-python sounddevice kokoro gradio-client

import cv2
from threading import Thread
from kokoro import KPipeline
from vqa_client import run_vqa  # now uses your gradio client logic

#def speak(text, pipeline):
#    for _, _, audio in pipeline(text, voice='ef_dora'):
#        pass  # Autoplay


#def speak(text, pipeline):
#    print("🔊 Speaking...")
#    for name, meta, audio in pipeline(text, voice='ef_dora'):
#        print(f"Audio info: {meta}")

import sounddevice as sd
import numpy as np


def speak(text, pipeline):
    for i, (gs, ps, audio) in enumerate(pipeline(text, voice='ef_dora', speed=1, split_pattern=r'\n+')):
        print(f"🔊 Speaking sentence {i+1}: {gs}")
        if isinstance(audio, (list, tuple)):  # Not expected, but just in case
            audio = np.array(audio, dtype=np.float32)
        sd.play(audio, samplerate=24000)
        sd.wait()  # Wait until this sentence finishes before playing the next


def main():
    tts = KPipeline(lang_code='e')  # Greek TTS

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("⚠️ Cannot open camera")
        return

    print("Καλησπέρα! Θα ρωτήσω: Τι βλέπεις; (Press 'q' to quit)")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️ Δεν μπορώ να διαβάσω το πλαίσιο")
            break

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Greek question
        question = "Τι βλέπεις;"
        answer = run_vqa(frame, question, greek=True)

        print(f"🗣️ Απάντηση: {answer}")
        Thread(target=speak, args=(answer, tts), daemon=True).start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

