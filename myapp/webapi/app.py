#!flask/bin/python
from flask import Flask
from flask import request
from flask import jsonify
import os
import io
import sys
import json
import numpy as np
import base64
from PIL import Image
import cv2
import torch
import random

import skimage.color
import skimage.io
import skimage.transform
from distutils.version import LooseVersion

from io import BytesIO

from PIL import Image, ImageDraw, ImageStat,ImageFont
from torchvision.transforms.functional import to_pil_image
import torch
import copy
import segmentation_models_pytorch as sm
import albumentations as albu

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)


np.set_printoptions(threshold=sys.maxsize)

global preprocessor
global model_2
global font
global DEVICE
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024

MODEL_NAME = "zhang_burn_wound"
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = "sigmoid"
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def load_model():
    global MODEL_NAME,preprocessor,model_2,font,ENCODER,ENCODER_WEIGHTS,DEVICE,ACTIVATION

    #load preprocessor
    preprocess_input = sm.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessor = get_preprocessing(preprocess_input)

    #load model
    model_folder_path = os.path.abspath("./models") + "/"
    DEVICE = torch.device("cpu")

    model_2_path = os.path.abspath(model_folder_path + "best_model_burn_deeplabv3plus_2.pth")

    if os.path.exists(model_2_path):
        model_2 = sm.DeepLabV3Plus(encoder_name=ENCODER,encoder_weights=ENCODER_WEIGHTS,classes=1,activation=ACTIVATION)
        model_2.load_state_dict(torch.load(model_2_path,map_location=torch.device("cpu")))
        model_2.eval()
    else:
        model_2 = None

    #load font
    font_folder_path = os.path.abspath("./webapi") + "/"
    font_path = os.path.abspath(font_folder_path + "arial.ttf")
    font = ImageFont.truetype(font_path, 25)

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

def resize_image(image, min_dim=1024, max_dim=1024, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1.0
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1.0, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = float(max_dim) / float(image_max)

    # Resize image using bilinear interpolation
    if scale != 1.0:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype)

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image






def predict_utils(model,image):

    #preprocess
    preprocess_image = preprocessor(image=image)['image']
    x_tensor = torch.from_numpy(preprocess_image).to(DEVICE).unsqueeze(0)
    pr_mask = model.forward(x_tensor).squeeze().cpu().detach().numpy().round()
    #pr_mask = np.transpose(pr_mask,(1,2,0))
    pixel_numbers = np.count_nonzero(pr_mask)

    return pr_mask,pixel_numbers

def merge_masks(image,masks1,cls1):
    for i in range(2):
        image = apply_mask(image,masks1[...,i],cls1[i])
    return image

def convert_image_to_base64(image):
    draw_image = Image.fromarray(image.astype('uint8'))

    buffer = BytesIO()
    draw_image.save(buffer, format='JPEG')
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return base64_image

def predict_image(image):
    global model_2,preprocessor

    image = np.asarray(image).astype(np.uint32)

    #image_np = resize_image(image,1024,1024)
    image_np = copy.deepcopy(image)
    image1 = copy.deepcopy(image_np)
    image2 = copy.deepcopy(image_np)
    pixels1 = {}#2 class

    if model_2 is not None:
        #reep ulceration
        masks1,pixels1 = predict_utils(model_2,image1)
        image2 = merge_masks(image2,masks1,[(255,0,0)])

    return image2,pixels1



load_model()
#/v1/model_ps/zhang_burn_wound:predict
@app.route('/v<int:version>/model_burn_deeplab/<string:model_name>:<string:action>', methods=['POST'])
def predict(version:int, model_name:str, action:str):
    if version != 1 or model_name != MODEL_NAME or action != "predict":
        return {"Not implemented", 501}

    jdata = request.data
    buffer = BytesIO(base64.b64decode(jdata))
    im_pil = Image.open(buffer).convert("RGB")

    result1,pixels1  = predict_image(im_pil)
    result1 = convert_image_to_base64(result1)
    result = {"result_image":result1,"result_pixel":pixels1}
    json_str = json.dumps(result)
    return json_str

@app.route('/', methods=['POST','GET'])
def index():
    return "WTF"
