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


ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

import mirle_vision


from mirle_vision.lib.bbox import BBox
from mirle_vision.lib.checkpoint import Checkpoint
from mirle_vision.lib.extension.functional import denormalize_means_stds
from mirle_vision.lib.task import Task
from mirle_vision.lib.task.instance_segmentation.palette import Palette

from mirle_vision.lib.task.instance_segmentation.checkpoint import Checkpoint
from mirle_vision.lib.task.instance_segmentation.model import Model
from mirle_vision.lib.task.instance_segmentation.preprocessor import Preprocessor
from mirle_vision.lib.task.instance_segmentation.inferer import Inferer

np.set_printoptions(threshold=sys.maxsize)

global inferer
global lower_prob_thresh
global upper_prob_thresh
global preprocessor
global model
global font
app = Flask(__name__)

MODEL_NAME = "zhang_pressure_sore"


def load_model():
    global inferer,lower_prob_thresh,upper_prob_thresh,MODEL_NAME,preprocessor,model,font
    model_folder_path = os.path.abspath("./models") + "/"
    device = torch.device("cpu")
    device_ids=[0]
    checkpoint = Checkpoint.load(os.path.abspath(model_folder_path + MODEL_NAME +".pth"),device)
    model = checkpoint.model
    preprocessor = model.preprocessor
    inferer = Inferer(model, device_ids)
    lower_prob_thresh = 0.8
    upper_prob_thresh = 1.0

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

def resize_image(image, min_dim=512, max_dim=512, min_scale=None, mode="square"):
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

def predict_image(image):
    global inferer,lower_prob_thresh,upper_prob_thresh,model,preprocessor,font
    processed_image, process_dict = preprocessor.process(image, is_train_or_eval=False)
    inference = \
        inferer.infer(image_batch=[processed_image],
                      lower_prob_thresh=lower_prob_thresh,
                      upper_prob_thresh=upper_prob_thresh)


    flatten_palette = Palette.get_flatten_palette()
    final_detection_bboxes = inference.final_detection_bboxes_batch[0]
    final_detection_classes = inference.final_detection_classes_batch[0]
    final_detection_probs = inference.final_detection_probs_batch[0]
    final_detection_probmasks = inference.final_detection_probmasks_batch[0]

    final_detection_bboxes = Preprocessor.inv_process_bboxes(process_dict, final_detection_bboxes)
    final_detection_probmasks = Preprocessor.inv_process_probmasks(process_dict, final_detection_probmasks)

    final_detection_bboxes = final_detection_bboxes.tolist()
    final_detection_categories = [model.class_to_category_dict[cls] for cls in final_detection_classes.tolist()]
    final_detection_probs = final_detection_probs.tolist()
    final_detection_probmasks = final_detection_probmasks.cpu()

    final_detection_colors = []
    final_detection_areas = []
    final_detection_polygon_group = []
    mask_image = torch.zeros((1, image.height, image.width), dtype=torch.uint8)
    contoured_mask_image = torch.zeros((1, image.height, image.width), dtype=torch.uint8)
    for i, probmask in enumerate(final_detection_probmasks):
        color = i + 1
        mask = (probmask > 0.5).byte()

        contours, _ = cv2.findContours(image=np.ascontiguousarray(mask.numpy().transpose(1, 2, 0)),
                                       mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)
        simple_contours, _ = cv2.findContours(image=np.ascontiguousarray(mask.numpy().transpose(1, 2, 0)),
                                              mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in simple_contours:
            epsilon = cv2.arcLength(curve=contour, closed=True) * 0.001
            polygon = cv2.approxPolyDP(curve=contour, epsilon=epsilon, closed=True)
            polygons.append([tuple(point) for point in polygon.squeeze(axis=1).tolist()])

        final_detection_colors.append(color)
        final_detection_areas.append(mask.sum().item())
        final_detection_polygon_group.append(polygons)
        mask_image |= mask * color
        contoured_mask_image |= mask * color

        for point in [point[0] for contour in contours for point in contour]:  # contour has shape (N, 1, 2)
            contoured_mask_image[:, point[1], point[0]] = 255  # take last index of instance for contour

    mask_image = to_pil_image(mask_image).convert('P')
    mask_image.putpalette(flatten_palette)
    contoured_mask_image = to_pil_image(contoured_mask_image).convert('P')
    contoured_mask_image.putpalette(flatten_palette)

    is_bright = ImageStat.Stat(image.convert('L')).rms[0] > 127
    offset = 0 if is_bright else 128
    category_to_color_dict = {category: tuple(random.randrange(0 + offset, 128 + offset) for _ in range(3))
                              for category in set(final_detection_categories)}

    """
    # region ===== Frame 1: Floating Mask =====
    draw_image = Image.new(mode='RGB', size=image.size)
    for probmask in final_detection_probmasks:
        probmask_image = to_pil_image(probmask)
        draw_image.paste(probmask_image, mask=probmask_image)
    # endregion ===================================================

    # region ===== Frame 2: Final Detection and Mask =====
    draw_image = Image.blend(draw_image.convert('RGBA'),
                             mask_image.convert('RGBA'),
                             alpha=0.8).convert('RGB')"""
    draw_image = image.copy()
    image_np = np.asarray(draw_image).astype(np.uint32)
    mask = np.asarray(mask_image).astype(np.uint32)
    alpha = 0.7
    color = [1,0,0]
    for c in range(3):
        image_np[:, :, c] = np.where(mask == 1,
                                  image_np[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image_np[:, :, c])

    draw_image = Image.fromarray(image_np.astype('uint8'))
    # endregion ==========================================

    # region ===== Frame 3: Final Detection and Mask with Contour and Area =====

    draw = ImageDraw.Draw(draw_image)
    for bbox, category, prob, area in zip(final_detection_bboxes, final_detection_categories,
                                          final_detection_probs, final_detection_areas):
        bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
        color = category_to_color_dict[category]
        text = '[{:d}] {:s} {:.3f}, {:d} pixels'.format(model.category_to_class_dict[category],
                                                        category if category.isascii() else '',
                                                        prob, area)
        draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color, width=4)
        draw.rectangle(((bbox.left, bbox.top + 30), (bbox.left + 12 * len(text), bbox.top)), fill=color)


        #draw.text((5, 5), char, (0,0,0),font=font)
        draw.text((bbox.left, bbox.top), text,font=font)
    # endregion ================================================================
    image_np = np.asarray(draw_image).astype(np.uint32)
    image_np = resize_image(image_np)
    draw_image = Image.fromarray(image_np.astype('uint8'))

    buffer = BytesIO()
    draw_image.save(buffer, format='JPEG')
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return base64_image




load_model()

@app.route('/v<int:version>/models/<string:model_name>:<string:action>', methods=['POST'])
def predict(version:int, model_name:str, action:str):
    if version != 1 or model_name != MODEL_NAME or action != "predict":
        return {"Not implemented", 501}

    jdata = request.data
    buffer = BytesIO(base64.b64decode(jdata))
    im_pil = Image.open(buffer).convert("RGB")

    result = {"mirle_result":predict_image(im_pil)}
    json_str = json.dumps(result)
    return json_str
@app.route('/', methods=['POST','GET'])
def index():
    if request.method=="GET":
        return "This is get request"
