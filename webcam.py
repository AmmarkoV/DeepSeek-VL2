# python3 -m venv venv
# source venv/bin/activate
# python3 -m pip install opencv-python sounddevice kokoro gradio-client

import cv2
import argparse
import subprocess
from threading import Thread
from kokoro import KPipeline
from vqa_client import run_vqa  # now uses your gradio client logic

import sounddevice as sd
import numpy as np


def translate_greek_to_english(text):
    result = subprocess.run(
        ['argos-translate', '--from', 'el', '--to', 'en', text],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def speak(text, pipeline, voice):
    for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')):
        print(f"🔊 Speaking sentence {i+1}: {gs}")
        if isinstance(audio, (list, tuple)):  # Not expected, but just in case
            audio = np.array(audio, dtype=np.float32)
        sd.play(audio, samplerate=24000)
        sd.wait()  # Wait until this sentence finishes before playing the next


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question', nargs='?', default=None, help='Question in Greek')
    args = parser.parse_args()

    if args.question:
        greek_question = args.question
        english_question = translate_greek_to_english(greek_question)
        print(f"Question: {greek_question} / {english_question}")
    else:
        greek_question = "Τι βλέπεις;"
        english_question = "What do you see?"

    ttsGreek = KPipeline(lang_code='e')  # Greek TTS
    ttsEnglish = KPipeline(lang_code='a')  # Greek TTS

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("⚠️ Cannot open camera")
        return

    print(f"Καλησπέρα! Θα ρωτήσω: {greek_question} (Press 'q' to quit)")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️ Δεν μπορώ να διαβάσω το πλαίσιο")
            break


        key = cv2.waitKey(1) & 0xFF
        if  key == ord('g'):
            answer = run_vqa(frame, greek_question, greek=True)
            print(f"🗣️ Απάντηση: {answer}")
            Thread(target=speak, args=(answer, ttsGreek,'ef_dora'), daemon=True).start()
        if  key == ord('e'):
            answer = run_vqa(frame, english_question, greek=False)
            print(f"🗣️ Answer: {answer}")
            Thread(target=speak, args=(answer, ttsEnglish,'af_bella'), daemon=True).start()

        # Overlay instruction text on the frame
        cv2.putText(
            frame, "Press 'G/E' to describe (in Greek or English), 'Q' to quit",
            (10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2, cv2.LINE_AA )
        cv2.imshow('Webcam', frame)

        if key  == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

