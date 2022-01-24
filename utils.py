import torch
import numpy as np
from typing import List
import torch.nn.functional as F
import torchvision
import os


# logging utils

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(preds, dim=1)
    dataset_size = preds.size()[0]
    return 100 * torch.sum(preds == labels).item()/dataset_size


def print_log(log: dict):
    for key in log.keys():
        print(f"{key}: {log[key]}")


def resize(image: torch.Tensor, min_limit: int = 224, max_limit: int = 224) -> torch.Tensor:
    min_size = float(min(image.shape[-2:]))
    max_size = float(max(image.shape[-2:]))
    scale_factor = min(
        min_limit / min_size,
        max_limit / max_size
    )

    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor,
        mode='bilinear', align_corners=False,
        recompute_scale_factor=True
    )[0]
    return image


# torch utils
def tensor_to_png(image: torch.Tensor, filename: str = "image"):
    to_pil_image = torchvision.transforms.ToPILImage()
    pil_image = to_pil_image(image)
    pil_image.save(str(os.path.join('some_images', filename + '.png')))


def to_torch_batch(image_list: List[torch.Tensor], device):
    images = torch.stack(image_list)
    return images.to(device)


def pad_img(image: torch.Tensor, pad_to: int = 224) -> torch.Tensor:
    """
    Add padding to the image to resize it to be 224x224
    """
    padding = [0, pad_to -
               image.size()[2], 0, pad_to - image.size()[1], 0, 0]
    resize_img = F.pad(image, padding)
    return resize_img


def resize_and_pad(image: torch.Tensor, min_limit: int = 224, max_limit: int = 224, pad_to: int = 224) -> torch.Tensor:
    image = resize(image, min_limit, max_limit)
    image = pad_img(image, pad_to)
    return image


def clean_batch(subject_images: List[torch.Tensor], verb_images: List[torch.Tensor], object_images: List[torch.Tensor], subject_labels: List[torch.Tensor], verb_labels: List[torch.Tensor], object_labels: List[torch.Tensor], keep_novel_subject: bool = False, min_limit: int = 224, max_limit: int = 224, pad_to: int = 224):
    # TODO: consider removing other novel categories (verbs, objects)
    '''
    Mutatively remove all of the none types from the batches and add padding to 
    fit 224x224 image size for the resent18 backbone
    '''
    assert len(subject_images) == len(verb_images) == len(object_images) == len(
        subject_labels) == len(verb_labels) == len(object_labels)
    i = 0
    while i < len(subject_images):
        remove = subject_images[i] is None or verb_images[i] is None or object_images[i] is None

        # since novel category is 0
        if not keep_novel_subject:
            remove = remove or subject_labels[i] == 0
            if subject_labels[i] is not None:
                subject_labels[i] -= 1

        if remove:
            subject_images.pop(i)
            verb_images.pop(i)
            object_images.pop(i)
            subject_labels.pop(i)
            verb_labels.pop(i)
            object_labels.pop(i)
        else:
            subject_images[i] = resize_and_pad(
                subject_images[i], min_limit, max_limit, pad_to)
            verb_images[i] = resize_and_pad(
                verb_images[i], min_limit, max_limit, pad_to)
            object_images[i] = resize_and_pad(
                object_images[i], min_limit, max_limit, pad_to)
            i += 1
