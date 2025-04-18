# Historical Document Text Restoration

This project focuses on restoring and enhancing the text from historical documents and images using Optical Character Recognition (OCR) and AI-powered text reconstruction. The tool allows users to upload scanned images of faded or damaged documents, extracts the text using OCR, and then enhances it using a generative AI model (Gemini 1.5). The project provides a web-based interface built with Gradio, offering an interactive and user-friendly experience.

## Features

- **Image Preprocessing**: Automatically converts images to grayscale for better text extraction.
- **OCR Text Extraction**: Uses Tesseract OCR to extract text from scanned images of documents.
- **AI Text Enhancement**: Employs the Gemini 1.5 AI model to reconstruct and enhance the text from the extracted OCR.
- **Accuracy Metrics**: Compares the OCR and AI-enhanced text with the provided correct transcription (if available), and calculates the accuracy.
- **User-Friendly Interface**: A simple and intuitive web interface built with Gradio for seamless interaction.

## Technologies Used

- **Frontend**: Gradio (for interactive web interface)
- **Backend**: Python (for image processing, OCR, and AI model integration)
- **OCR**: Tesseract OCR
- **Generative AI**: Gemini 1.5
- **Image Processing**: OpenCV (for image manipulation and preprocessing)
- **Version Control**: Git

## Setup & Installation

### Requirements

- Python 3.8 or later
- Tesseract OCR
- Gemini 1.5 API Key
- Gradio
- OpenCV
- Other Python dependencies listed in `requirements.txt`

### Installation Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/historical-document-text-restoration.git
    cd historical-document-text-restoration
    ```

2. **Install dependencies:**

    - Create a virtual environment (recommended):

        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```

    - Install the required dependencies:

        ```bash
        pip install -r requirements.txt
        ```

3. **Set up Tesseract OCR:**

    - **Windows**: Download and install Tesseract from [here](https://github.com/UB-Mannheim/tesseract/wiki).
    - **Linux**: Install using:

        ```bash
        sudo apt-get install tesseract-ocr
        ```

4. **Get Gemini 1.5 API Key:**

    - Register on the [Gemini API platform](https://genai.com) to get your API key.
    - Create a `.env` file in the project root and add your API key:

        ```bash
        GEMINI_API_KEY=your_api_key_here
        ```

5. **Run the application:**

    ```bash
    python app.py
    ```

    This will start a local web server, and you can open the interface by navigating to `http://localhost:7860` in your browser.

## Usage

1. Upload a scanned image of a historical document.
2. The app will automatically preprocess the image and run OCR to extract the text.
3. If available, enter the correct transcription of the document.
4. The tool will enhance the OCR text using AI and display the accuracy metrics and enhanced text.
5. You can view the processed image and OCR text in the corresponding tabs.

## Contributing

We welcome contributions to improve this project! To contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and commit them.
4. Open a pull request to the main repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Tesseract OCR**: Open-source OCR tool.
- **Gemini 1.5**: Generative AI model used for enhancing text.
- **Gradio**: Used for creating the web-based user interface.
- **OpenCV**: Used for image processing tasks.

## Contact

If you have any questions or feedback, feel free to reach out at [your-email@example.com](mailto:your-email@example.com).
