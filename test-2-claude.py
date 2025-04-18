import gradio as gr
import cv2
import numpy as np
import pytesseract
import re
import google.generativeai as genai
from rapidfuzz.distance import Levenshtein
import os
import platform
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

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
API_KEY = os.getenv("GEMINI_API_KEY") # Replace with your actual API key
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
def threshold_image(img, threshold_value=None):
    """Apply thresholding to image with caching"""
    cache_key = f"threshold_{id(img)}_{threshold_value}"
    if cache_key in image_cache:
        return image_cache[cache_key]
    
    if threshold_value is None:  # Adaptive thresholding
        thresholded_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
    else:  # Manual thresholding
        _, thresholded_image = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    
    image_cache[cache_key] = thresholded_image
    return thresholded_image

def bm3d_denoising(img, sigma_psd=55):
    """Apply BM3D denoising with caching"""
    cache_key = f"bm3d_{id(img)}_{sigma_psd}"
    if cache_key in image_cache:
        return image_cache[cache_key]
    
    denoised_image = cv2.fastNlMeansDenoising(img, None, sigma_psd)
    image_cache[cache_key] = denoised_image
    return denoised_image

def remove_noise(img, kernel_size=3):
    """Apply simple noise removal with caching"""
    cache_key = f"denoise_{id(img)}_{kernel_size}"
    if cache_key in image_cache:
        return image_cache[cache_key]
    
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    denoised = cv2.filter2D(img, -1, kernel)
    result = cv2.medianBlur(denoised, 3)
    
    image_cache[cache_key] = result
    return result

def sharpen_image(img):
    """Apply sharpening with caching"""
    cache_key = f"sharpen_{id(img)}"
    if cache_key in image_cache:
        return image_cache[cache_key]
    
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    
    image_cache[cache_key] = sharpened
    return sharpened

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

def parallel_process_image(img):
    """Process image with parallel OCR"""
    # Process the image
    thresholded = threshold_image(img, 242)  # Default threshold
    bm3d_denoised_image = bm3d_denoising(thresholded)
    denoised = remove_noise(thresholded)
    sharpened_image = sharpen_image(bm3d_denoised_image)
    
    # Perform OCR in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        ocr_tasks = {
            'original': executor.submit(perform_ocr, img),
            'thresholded': executor.submit(perform_ocr, thresholded),
            'bm3d_denoised': executor.submit(perform_ocr, bm3d_denoised_image),
            'denoised': executor.submit(perform_ocr, denoised),
            'sharpened': executor.submit(perform_ocr, sharpened_image)
        }
        
        # Get results and clean text
        original_text = remove_extra_spaces_and_lines(ocr_tasks['original'].result())
        thresholded_text = remove_extra_spaces_and_lines(ocr_tasks['thresholded'].result())
        bm3d_denoised_text = remove_extra_spaces_and_lines(ocr_tasks['bm3d_denoised'].result())
        denoised_text = remove_extra_spaces_and_lines(ocr_tasks['denoised'].result())
        sharpened_text = remove_extra_spaces_and_lines(ocr_tasks['sharpened'].result())
    
    return {
        'images': {
            'original': img,
            'thresholded': thresholded,
            'bm3d_denoised': bm3d_denoised_image,
            'denoised': denoised,
            'sharpened': sharpened_image
        },
        'texts': {
            'original': original_text,
            'thresholded': thresholded_text,
            'bm3d_denoised': bm3d_denoised_text,
            'denoised': denoised_text,
            'sharpened': sharpened_text
        }
    }

def get_gemini_response(texts):
    """Get improved text from Gemini API"""
    if not model:
        return "Error: Gemini model not available", {}
    
    for key, text in texts.items():
        if text:
            logging.info(f"{key.capitalize()} text: {text[:100]}...")
        else:
            logging.warning(f"{key.capitalize()} text is empty")
    
    if all(not text for text in texts.values()):
        return "No text detected in any processed image. Try adjusting image settings or using a clearer image.", {}
    
    # Enhanced prompt with examples of word parts
    user_prompt = f"""
    Task: Restore faded text from multiple OCR results of the same document.
    
    I have a historical document with faded text, and I've applied different image processing techniques before OCR. 
    Please analyze all these OCR outputs and reconstruct the most likely original text.
    
    OCR RESULTS:
    1. Original image: "{texts['original']}"
    2. Thresholded image: "{texts['thresholded']}"
    3. BM3D denoised image: "{texts['bm3d_denoised']}"
    4. Basic denoised image: "{texts['denoised']}"
    5. Sharpened image: "{texts['sharpened']}"
    
    Important:
    - Look for consistent patterns across all results
    - When one OCR method catches parts of words missing in others, incorporate them
    - Fix obvious OCR errors (like '0' instead of 'O')
    - Maintain original formatting where possible
    
    Return ONLY the fully reconstructed text without any explanations, headers, or notes.
    """
    
    try:
        response = model.generate_content(user_prompt)
        return response.text, texts
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return f"Error generating response: {str(e)}", texts

def process_image(image, threshold_value=None, correct_transcription=None):
    """Main image processing function"""
    if image is None:
        return [None]*11 + ["No image provided."]
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image
        
    # Apply custom threshold if specified
    if threshold_value is not None:
        # Clear the cache when threshold changes
        global image_cache
        image_cache = {}
    
    # Process image and get OCR results
    results = parallel_process_image(img)
    
    # Get Gemini model response
    model_text, texts = get_gemini_response(results['texts'])
    
    # If correct transcription is not provided, use model response
    if not correct_transcription:
        if model_text and not model_text.startswith("Error:"):
            correct_transcription = model_text
        else:
            correct_transcription = ""

    # Calculate accuracy metrics
    accuracies = {}
    accuracy_metrics = "No accurate comparison available."
    
    if correct_transcription:
        for key, text in texts.items():
            accuracies[key] = calculate_accuracy(text, correct_transcription)
        
        if model_text and not model_text.startswith("Error:"):
            accuracies['model'] = calculate_accuracy(model_text, correct_transcription)
        else:
            accuracies['model'] = 0.0
            
        accuracy_metrics = "\n".join([ 
            f"{key.capitalize()} Image Accuracy: {acc:.2%}" 
            for key, acc in accuracies.items()
        ])

    # Return all results
    return (
        image, 
        results['images']['thresholded'],
        results['images']['bm3d_denoised'],
        results['images']['denoised'],
        results['images']['sharpened'],
        results['texts']['original'],
        results['texts']['thresholded'],
        results['texts']['bm3d_denoised'],
        results['texts']['denoised'],
        results['texts']['sharpened'],
        model_text,
        accuracy_metrics
    )

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Historical Document Text Restoration")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ## Instructions
            1. Upload an image with faded text
            2. Adjust threshold or use adaptive thresholding
            3. Click "Process Image" to extract and restore text
            4. Optional: Provide correct transcription to calculate accuracy
            """)
            
            # Input controls
            image_input = gr.Image(label="Upload Document Image", type="numpy")
            
            with gr.Row():
                threshold_slider = gr.Slider(
                    label="Threshold Value", 
                    minimum=0, 
                    maximum=255, 
                    step=1, 
                    value=242
                )
                adaptive_checkbox = gr.Checkbox(
                    label="Use Adaptive Thresholding", 
                    value=False
                )
                
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
                
                with gr.TabItem("Processed Images"):
                    image_tabs = gr.Tabs()
                    with image_tabs:
                        with gr.TabItem("Original"):
                            original_image_display = gr.Image(label="Original Image")
                            original_text_display = gr.Textbox(label="OCR Text", lines=4)
                        with gr.TabItem("Thresholded"):
                            thresholded_image_display = gr.Image(label="Thresholded Image")
                            thresholded_text_display = gr.Textbox(label="OCR Text", lines=4)
                        with gr.TabItem("BM3D Denoised"):
                            bm3d_denoised_image_display = gr.Image(label="BM3D Denoised Image")
                            bm3d_denoised_text_display = gr.Textbox(label="OCR Text", lines=4)
                        with gr.TabItem("Basic Denoised"):
                            denoised_image_display = gr.Image(label="Denoised Image")
                            denoised_text_display = gr.Textbox(label="OCR Text", lines=4)
                        with gr.TabItem("Sharpened"):
                            sharpened_image_display = gr.Image(label="Sharpened Image")
                            sharpened_text_display = gr.Textbox(label="OCR Text", lines=4)
            
    process_button.click(
        process_image, 
        inputs=[image_input, threshold_slider, transcription_input],
        outputs=[
            original_image_display,
            thresholded_image_display,
            bm3d_denoised_image_display,
            denoised_image_display,
            sharpened_image_display,
            original_text_display,
            thresholded_text_display,
            bm3d_denoised_text_display,
            denoised_text_display,
            sharpened_text_display,
            model_text_display,
            accuracy_output
        ]
    )

    clear_button.click(lambda: None, outputs=[image_input, transcription_input, model_text_display, accuracy_output])

demo.launch()
