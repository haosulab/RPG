import cv2
import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image  
import sys
import pydoc


def arr_to_str(arr, precision=4):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    s = "["
    for i in arr:
        s += f"{i:>{precision+3}.{precision}f}, "
    s += "]"
    return s


# def write_text(img, text):
#     if text is not None:
#         font                   = cv2.FONT_HERSHEY_SIMPLEX
#         bottomLeftCornerOfText = (10, 0)
#         fontScale              = 1
#         fontColor              = (255, 255, 255)
#         thickness              = 2
#         lineType               = cv2.LINE_AA

#         lines = text.split("\n")

#         for i, line in enumerate(lines):
#             bottomLeftCornerOfText = (
#                 bottomLeftCornerOfText[0],
#                 bottomLeftCornerOfText[1] + 30,
#             )
#             img = cv2.putText(
#                 img.astype(np.uint8).copy(),
#                 line, 
#                 bottomLeftCornerOfText, 
#                 font, 
#                 fontScale,
#                 fontColor,
#                 thickness,
#                 lineType)
#     return img

def write_text(img, text, fill=(255, 255, 255), fontsize=24):
    if text is not None:
        bottomLeftCornerOfText = (10, 10)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        pil_im = Image.fromarray(img_rgb)  
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",fontsize)
        draw.text(bottomLeftCornerOfText, text, fill=fill, font=font)
        img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return img


def exp_coord2angle_axis(expcoord):
    norm = np.linalg.norm(expcoord)
    if norm == 0:
        theta = 0
        omega = np.array([1, 0, 0])
    else:
        omega = expcoord / norm
        theta = norm
    return theta, omega


def output_help_to_file(filepath, request):
    f = open(filepath, 'w')
    sys.stdout = f
    pydoc.help(request)
    f.close()
    sys.stdout = sys.__stdout__
    return