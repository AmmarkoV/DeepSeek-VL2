# vqa_client.py
import tempfile
import cv2
import numpy as np
from gradio_client import Client, handle_file

# Server configuration
client = Client("http://139.91.185.16:8083")  # Change if needed
#client = Client("http://147.52.17.119:8083")  # Change if needed
 

def run_vqa(image: np.ndarray, question: str, greek=False):
    # Save image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        filename = tmp.name
        cv2.imwrite(filename, image)

    try:
        imageFile = handle_file(filename)
        imageList = [imageFile]

        # Reset session
        client.predict(api_name="/reset_state")

        # Transfer input
        client.predict(
            input_images=imageList,
            input_text=question,
            api_name="/transfer_input"
        )

        # Get prediction
        result = client.predict(
            chatbot=[],
            temperature=0.6,
            top_p=0.9,
            max_length_tokens=100,
            repetition_penalty=1.1,
            max_context_length_tokens=4096,
            greek_translation=greek,
            api_name="/predict"
        )

        return result[0][0][1]
    except Exception as e:
        print(f"[run_vqa] Error during VQA inference: {e}")
        return "⚠️ Σφάλμα κατά την επεξεργασία"

