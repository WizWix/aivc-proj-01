import sys
import os
import urllib.request
import logging

try:
    import easyocr
except ImportError:
    print("Error: easyocr is not installed.")
    print("Please install it using: pip install easyocr")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Default image with Korean text (EasyOCR example)
DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/korean.png"
DEFAULT_IMAGE_FILENAME = "sample_korean_text.png"

def download_sample_image(filename=DEFAULT_IMAGE_FILENAME):
    """Downloads a sample image containing Korean characters if it doesn't exist."""
    if not os.path.exists(filename):
        logging.info(f"Sample image not found. Downloading from internet...")
        try:
            req = urllib.request.Request(
                DEFAULT_IMAGE_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
            with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                out_file.write(response.read())
            logging.info(f"Downloaded sample image to {filename}")
        except Exception as e:
            logging.error(f"Failed to download image: {e}")
            sys.exit(1)
    return filename

def get_ocr_reader():
    """Initializes and returns the EasyOCR reader for Korean and English."""
    # Initialize the reader only once, as it loads models into memory
    # gpu=True uses GPU if available, else falls back to CPU
    logging.info("Initializing EasyOCR reader for Korean and English...")
    reader = easyocr.Reader(['ko', 'en'])
    return reader

def extract_text(image_path):
    """
    Extracts Korean and English text from an image.
    This function can be easily imported and used in a FastAPI route later.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    reader = get_ocr_reader()
    
    logging.info(f"Running OCR on {image_path}...")
    # readtext returns a list of tuples: (bounding_box, text, confidence)
    results = reader.readtext(image_path)
    
    return results

def print_results(results):
    """Formats and prints the OCR results."""
    if not results:
        logging.warning("No text detected in the image.")
        return

    print("\n--- OCR Results ---")
    for (bbox, text, prob) in results:
        print(f"Detected Text: {text} | Confidence: {prob:.4f}")
    print("-------------------\n")

def main():
    # If a path is provided as a commandline argument, use it.
    # Otherwise, download and use the default sample image.
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        logging.info("No image file provided. Using default sample image.")
        # Put the image in the current directory
        # We could also put it in an 'images' directory
        images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, DEFAULT_IMAGE_FILENAME)
        download_sample_image(image_path)

    try:
        results = extract_text(image_path)
        print_results(results)
    except Exception as e:
        logging.error(f"OCR processing failed: {e}")

if __name__ == "__main__":
    main()
