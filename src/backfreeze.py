from fastapi import FastAPI
from pydantic import BaseModel
from pyzbar import pyzbar
import PIL
import pytesseract
import cv2
import base64
import io
import numpy as np

def read_image(base64_image):
    pytesseract.pytesseract.tesseract_cmd = "/app/.apt/usr/bin/tesseract"
    # Convert the image to grayscale a
    imgdata = base64.b64decode(base64_image)
    img = PIL.Image.open(io.BytesIO(imgdata))

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    out_gray=cv2.divide(img, bg, scale=255)
    img = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 


    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=4)
    #img = cv2.fastNlMeansDenoising(img, None, 20, 7, 21) 
    #img = cv2.erode(img, kernel, iterations=2)
    
    text = pytesseract.image_to_string(img, config="-c tessedit_char_whitelist=0123456789./-")
    text = text.strip()
    text = text.replace(" ", "")
    print(text)
    return text

def read_barcode(base64_image):
    imgdata = base64.b64decode(base64_image)
    img = PIL.Image.open(io.BytesIO(imgdata))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    decoded = pyzbar.decode(img)
    print(decoded)
    return decoded[0].data.decode("utf-8")

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

@app.post("/barcode/")
async def create_item(item: Image):
    """
    Reads the image and returns the barcode data
    """
    text = read_barcode(item.base64_img)
    return {'text_str': text, 'Status': 'success'}