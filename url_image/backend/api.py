import base64
import io
import logging
import sys
import numpy as np
import requests
from imageio import imsave
from PIL import Image
import PIL
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
from torchvision import transforms

from .models import unet_resnext50

def sanitize(state_dict):
    cpu = torch.device('cpu')
    sanitized = dict()
    for key in state_dict:
        if key.startswith('module.'):
            sanitized[key[7:]] = state_dict[key].to(cpu)
        else:
            sanitized[key] = state_dict[key].to(cpu)
    return sanitized


def load_state(path):
    state = torch.load(path, map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    state = sanitize(state)
    return state


class Segmentator(object):
    size = (320, 240)

    meanstd = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    normalize = transforms.Normalize(**meanstd)
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.Pad((8, 0), padding_mode='reflect'),
        transforms.ToTensor(),
        normalize
    ])

    def __init__(self):
        self.net = unet_resnext50(num_classes=1, pretrained=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @torch.no_grad()
    def predict(self, image):
        image = self.preprocess(image)
        tensor = torch.stack((image,)).to(self.device)
        logits = self.net(tensor)
        probs = torch.sigmoid(logits).data[0, 0, :, 8:-8].to('cpu').numpy()
        return probs



def get_img_and_base64(url):
    """
    returns image in base64 format (a string)
    that can be passed to html as context and rendered without saving to drive
    """
    blob = io.BytesIO( requests.get(url).content )
    img = Image.open(blob).convert('RGB')
    img_np = np.array(img).astype(np.uint8)
    fmem = io.BytesIO()
    imsave(fmem, img_np, 'png')
    fmem.seek(0)
    img64 = base64.b64encode(fmem.read()).decode('utf-8')
    return (img, img64)
 

def get_person_mask_and_mask64(img):
    # LOGGING_LEVEL = 'INFO'
    # LOGGING_FORMAT = '[%(asctime)s] %(name)s:%(lineno)d: %(message)s'
    # logging.basicConfig(format=LOGGING_FORMAT, level=LOGGING_LEVEL)
    # logger = logging.getLogger(__file__)
    segmentator = Segmentator()
    mask_np = segmentator.predict(img)
    mask_np = (mask_np * 255).astype(np.uint8)
    mask = Image.fromarray(mask_np)

    # img_np = np.array(img)
    # size = (img_np.shape[1], img_np.shape[0])
    reshaped_mask = mask.resize(img.size, PIL.Image.BILINEAR)

    fmem = io.BytesIO()
    imsave(fmem, reshaped_mask, 'png')
    fmem.seek(0)
    mask64 = base64.b64encode(fmem.read()).decode('utf-8')
    return (reshaped_mask, mask64)


def merge_style_and_person(mask, person, background):
    threshold_for_mask = 253

    back_w, back_h = background.size
    person_w, person_h = mask.size

    if float(back_w) / float(back_h) > float(person_w) / float(person_h):
        resized_person = person.resize( (int(back_h * float(person_w) / float(person_h)), back_h), PIL.Image.BILINEAR)
        resized_mask = mask.resize( (int(back_h * float(person_w) / float(person_h)), back_h), PIL.Image.BILINEAR)

    else:
        resized_person = person.resize( (back_w, int(float(person_h) / float(person_w) * back_w)), PIL.Image.BILINEAR)
        resized_mask = mask.resize( (back_w, int(float(person_h) / float(person_w) * back_w)), PIL.Image.BILINEAR)

    # print("resized_mask.size", resized_mask.size)
    # print("resized_person.size", resized_person.size)
    # print("background.size", background.size)

    mask_np = np.array(resized_mask)
    person_np = np.array(resized_person)
    back_np = np.array(background)

    # print("mask_np max and min", np.max(mask_np), np.min(mask_np))

    p_w, p_h = resized_person.size

    patch_under_person = back_np[back_h // 2 - p_h // 2 : back_h // 2 + p_h // 2 + p_h % 2, \
    back_w // 2 - p_w // 2 : back_w // 2 + p_w // 2 + p_w % 2, :]

    # print("patch_under_person.shape", patch_under_person.shape)

    red = np.where(mask_np > threshold_for_mask, person_np[:,:,0], patch_under_person[:,:,0])
    green = np.where(mask_np > threshold_for_mask, person_np[:,:,1], patch_under_person[:,:,1])
    blue = np.where(mask_np > threshold_for_mask, person_np[:,:,2], patch_under_person[:,:,2])

    # print("red.shape", red.shape)
    # print("regreend.shape", green.shape)
    # print("blue.shape", blue.shape)

    patch_under_person[:,:,0] = red
    patch_under_person[:,:,1] = green
    patch_under_person[:,:,2] = blue

    back_np[back_h // 2 - p_h // 2 : back_h // 2 + p_h // 2 + p_h % 2, \
    back_w // 2 - p_w // 2 : back_w // 2 + p_w // 2 + p_w % 2, :] = patch_under_person

    fmem = io.BytesIO()
    imsave(fmem, back_np, 'png')
    fmem.seek(0)
    merged64 = base64.b64encode(fmem.read()).decode('utf-8')
    return merged64
    # patch_under_person[:,:,0][mask_np > 0] = 0


