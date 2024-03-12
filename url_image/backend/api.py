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
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F

from .models import unet_resnext50


class Segmentator:
    size = (320, 320)
    step = 32
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, step: int = 32) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = unet_resnext50(num_classes=1, pretrained=True)
        self.net.eval()
        self.net.to(self.device)

        self.step = step
        self.pt = 0
        self.pr = 0
        self.pb = 0
        self.pl = 0

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image.thumbnail(self.size)

        tensor = F.to_tensor(image)

        h, w = tensor.shape[-2:]
        self.pl = (w % self.step + 1) // 2
        self.pt = (h % self.step) // 2
        self.pr = (w % self.step) // 2
        self.pb = (h % self.step + 1) // 2
        tensor = F.pad(
            tensor, [self.pl, self.pt, self.pr, self.pb], padding_mode="reflect"
        )

        tensor = F.normalize(tensor, self.mean, self.std)
        tensor = tensor.to(self.device)

        batch = torch.stack([tensor])

        return batch

    def postprocess(self, logits: torch.Tensor) -> np.ndarray:
        logits = torch.squeeze(logits)

        h, w = logits.shape
        logits = logits[self.pt : h - self.pb, self.pl : w - self.pr]

        probs = torch.sigmoid(logits)
        probs = probs.to("cpu")
        probs = probs.numpy()

        return probs

    @torch.no_grad()
    def predict(self, image: Image.Image) -> np.ndarray:
        batch = self.preprocess(image)
        logits = self.net(batch)
        probs = self.postprocess(logits)
        return probs


def get_img_and_base64(url):
    """
    returns image in base64 format (a string)
    that can be passed to html as contex§t and rendered without saving to drive
    """
    blob = io.BytesIO(requests.get(url).content)
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
        resized_person = person.resize((int(back_h * float(person_w) / float(person_h)), back_h), PIL.Image.BILINEAR)
        resized_mask = mask.resize((int(back_h * float(person_w) / float(person_h)), back_h), PIL.Image.BILINEAR)

    else:
        resized_person = person.resize((back_w, int(float(person_h) / float(person_w) * back_w)), PIL.Image.BILINEAR)
        resized_mask = mask.resize((back_w, int(float(person_h) / float(person_w) * back_w)), PIL.Image.BILINEAR)

    # print("resized_mask.size", resized_mask.size)
    # print("resized_person.size", resized_person.size)
    # print("background.size", background.size)

    mask_np = np.array(resized_mask)
    person_np = np.array(resized_person)
    back_np = np.array(background)

    # print("mask_np max and min", np.max(mask_np), np.min(mask_np))

    p_w, p_h = resized_person.size

    patch_under_person = back_np[back_h // 2 - p_h // 2: back_h // 2 + p_h // 2 + p_h % 2, \
                         back_w // 2 - p_w // 2: back_w // 2 + p_w // 2 + p_w % 2, :]

    # print("patch_under_person.shape", patch_under_person.shape)

    red = np.where(mask_np > threshold_for_mask, person_np[:, :, 0], patch_under_person[:, :, 0])
    green = np.where(mask_np > threshold_for_mask, person_np[:, :, 1], patch_under_person[:, :, 1])
    blue = np.where(mask_np > threshold_for_mask, person_np[:, :, 2], patch_under_person[:, :, 2])

    # print("red.shape", red.shape)
    # print("regreend.shape", green.shape)
    # print("blue.shape", blue.shape)

    patch_under_person[:, :, 0] = red
    patch_under_person[:, :, 1] = green
    patch_under_person[:, :, 2] = blue

    back_np[back_h // 2 - p_h // 2: back_h // 2 + p_h // 2 + p_h % 2, \
    back_w // 2 - p_w // 2: back_w // 2 + p_w // 2 + p_w % 2, :] = patch_under_person

    fmem = io.BytesIO()
    imsave(fmem, back_np, 'png')
    fmem.seek(0)
    merged64 = base64.b64encode(fmem.read()).decode('utf-8')
    return merged64
