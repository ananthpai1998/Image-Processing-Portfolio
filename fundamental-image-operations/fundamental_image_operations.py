import base64
import json
import sys
from io import BytesIO
from PIL import Image
import numpy as np
import scipy.ndimage as ndimage

def convert_image_to_grayscale(image_data):
    image = Image.open(BytesIO(image_data))
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image)
    return img_array

def opening_image(img_array):
    img_array = ndimage.binary_opening(img_array).astype(np.uint8) * 255
    img_array = Image.fromarray((img_array).astype('uint8'))
    buffered = BytesIO()
    img_array.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def closing_image(img_array):
    img_array = ndimage.binary_closing(img_array).astype(np.uint8) * 255
    img_array = Image.fromarray((img_array).astype('uint8'))
    buffered = BytesIO()
    img_array.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def dialation_image(img_array):
    img_array = ndimage.binary_dilation(img_array).astype(np.uint8) * 255
    img_array = Image.fromarray((img_array).astype('uint8'))
    buffered = BytesIO()
    img_array.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def errosion_image(img_array):
    selem = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=bool)  
    img_array = ndimage.binary_erosion(img_array > 0, structure=selem).astype(np.uint8) * 255
    img_array = Image.fromarray((img_array).astype('uint8'))
    buffered = BytesIO()
    img_array.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def main():
    try:
        input_data = sys.stdin.read()

        data = json.loads(input_data)
        image_data = base64.b64decode(data['data'])

        img_array = convert_image_to_grayscale(image_data)

        opening_image_arr = opening_image(img_array)
        closing_image_arr = closing_image(img_array)
        dialation_image_arr = dialation_image(img_array)
        errosion_image_arr = errosion_image(img_array)

        img = Image.fromarray((img_array).astype('uint8'))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")



        print(json.dumps( [{
            "title": "Original Image",
            "img": data['data'],
            "description": "",
            },
            {
            "title": "Grayscale Image",
            "img": img_str,
            "description": "",
            },
            {
            "title": "Opening Image",
            "img": opening_image_arr,
            "description": "",
            },
            {
            "title": "Closing Image",
            "img": closing_image_arr,
            "description": "",
            },
            {
            "title": "Dialation Image",
            "img": dialation_image_arr,
            "description": "",
            },
            {
            "title": "Errosion Image",
            "img": errosion_image_arr,
            "description": "",
            },
            ]))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        

if __name__ == "__main__":
    main()