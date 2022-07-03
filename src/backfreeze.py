from cv2 import imread
from fastapi import FastAPI
from pydantic import BaseModel
import PIL
import pytesseract
import cv2
import base64
import io
import numpy as np

def read_image(base64_image):
    pytesseract.pytesseract.tesseract_cmd = "/app/.apt/usr/bin/tesseract"
    # Convert the image to grayscale
    imgdata = base64.b64decode(base64_image)
    img = PIL.Image.open(io.BytesIO(imgdata))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1,1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    text = pytesseract.image_to_string(img)
    return text
  


class Image(BaseModel):
    base64_img: str = ""

app = FastAPI()

@app.post("/img_to_str/")
async def create_item(item: Image):
    """
    Reads the image and returns the text in the image.
    """
    text = read_image(item.base64_img)
    return {'text_str': text, 'Status': 'success'}