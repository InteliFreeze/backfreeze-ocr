from fastapi import FastAPI
from pydantic import BaseModel
import pytesseract
import cv2
import numpy as np

def read_image(rgb_array, height, width):
    pytesseract.pytesseract.tesseract_cmd = "/app/.apt/usr/bin/tesseract"
    # Convert the image to grayscale
    rgb_matrix = np.array(rgb_array).astype(np.uint8)
    rgb_3d = np.reshape(rgb_matrix, (int(height), int(width), 3))
    img = cv2.resize(rgb_3d, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((1,1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    text = pytesseract.image_to_string(img)
    return text
  


class Image(BaseModel):
    rgb_array: list[int] = [];
    height: int = None;
    width: int = None;

app = FastAPI()

@app.post("/img_to_str/")
async def create_item(item: Image):
    """
    Reads the image and returns the text in the image.
    """
    text = read_image(item.rgb_array, item.height, item.width)
    return {'text_str': text, 'Status': 'success'}