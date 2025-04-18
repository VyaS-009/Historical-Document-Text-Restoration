import gradio as gr
import cv2
import numpy as np
import pytesseract
import re
import google.generativeai as genai
import os
import platform
import logging
from dotenv import load_dotenv
from rapidfuzz.distance import Levenshtein

load_dotenv()
# Load your Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Tesseract OCR path
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        logging.warning(f"Tesseract not found at {pytesseract.pytesseract.tesseract_cmd}. Please verify installation.")
elif platform.system() == 'Linux':
    os.system('apt-get update && apt-get install -y tesseract-ocr')

# Configure Generative AI
API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with your actual API key
genai.configure(api_key=API_KEY)

try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    logging.info("Gemini model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Gemini model: {e}")
    model = None

# Cache for processed images to avoid redundant processing
image_cache = {}

# Image processing functions
def remove_extra_spaces_and_lines(text):
    """Clean up OCR text"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text

def calculate_accuracy(text1, text2):
    """Calculate text similarity using Levenshtein distance"""
    if not text1 or not text2:
        return 0.0
    
    distance = Levenshtein.distance(text1, text2)
    max_length = max(len(text1), len(text2))
    
    # Prevent division by zero
    if max_length == 0:
        return 0.0
        
    accuracy = (1.0 - (distance / max_length))
    return accuracy

def perform_ocr(image):
    """Run OCR on an image with error handling"""
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return ""

def process_image(image, correct_transcription=None):
    """Main image processing function"""
    if image is None:
        return [None]*7 + ["No image provided."]
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image
        
    # Perform OCR on the original image
    original_text = remove_extra_spaces_and_lines(perform_ocr(img))
    
    # Get Gemini model response
    if model:
        try:
            model_response = model.generate_content(f"Reconstruct the original text from this OCR result: {original_text}")
            model_text = model_response.text
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            model_text = ""
    else:
        model_text = ""
    
    # If correct transcription is not provided, use model response
    if not correct_transcription:
        if model_text and not model_text.startswith("Error:"):
            correct_transcription = model_text
        else:
            correct_transcription = ""

    # Calculate accuracy metrics
    accuracy_metrics = "No accurate comparison available."
    
    if correct_transcription:
        accuracy = calculate_accuracy(original_text, correct_transcription)
        
        if model_text and not model_text.startswith("Error:"):
            model_accuracy = calculate_accuracy(model_text, correct_transcription)
        else:
            model_accuracy = 0.0
            
        accuracy_metrics = f"Original Image Accuracy: {accuracy:.2%}\nModel Image Accuracy: {model_accuracy:.2%}"

    # Return all results
    return (
        image,  # Original Image
        original_text,  # OCR Text from Original Image
        img,  # Processed Image (grayscale version in this case)
        original_text,  # OCR Text on processed image
        model_text,  # Model-enhanced Text
        accuracy_metrics  # Accuracy metrics
    )

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Historical Document Text Restoration")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ## Instructions
            1. Upload an image with faded text
            2. Click "Process Image" to extract and restore text
            3. Optional: Provide correct transcription to calculate accuracy
            """)
            
            # Input controls
            image_input = gr.Image(label="Upload Document Image", type="numpy")
            transcription_input = gr.Textbox(
                label="Correct Transcription (Optional)", 
                placeholder="Enter the correct text if you have it"
            )
            
            process_button = gr.Button("Process Image", variant="primary")
            clear_button = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=2):
            tabs = gr.Tabs()
            with tabs:
                with gr.TabItem("Results"):
                    model_text_display = gr.Textbox(
                        label="Restored Text (AI-Enhanced)", 
                        lines=10,
                        show_copy_button=True
                    )
                    accuracy_output = gr.Textbox(
                        label="Accuracy Metrics", 
                        lines=6
                    )
                
                with gr.TabItem("Processed Image"):
                    processed_image_display = gr.Image(label="Processed Image")
                    ocr_text_display = gr.Textbox(label="OCR Text", lines=4)

    process_button.click(
        process_image, 
        inputs=[image_input, transcription_input],
        outputs=[
            processed_image_display,
            ocr_text_display,
            processed_image_display,
            ocr_text_display,
            model_text_display,
            accuracy_output
        ]
    )

    clear_button.click(lambda: None, outputs=[image_input, transcription_input, model_text_display, accuracy_output])

demo.launch()
