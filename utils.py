import torch
import numpy as np
from typing import List
import torch.nn.functional as F

RESNET_INPUT_SIZE = 224


# logging utils

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(preds, dim=1)
    dataset_size = preds.size()[0]
    return 100 * torch.sum(preds == labels).item()/dataset_size


def print_log(log: dict):
    for key in log.keys():
        print(f"{key}: {log[key]}")
    print()


# torch utils

def to_torch_batch(image_list: List[torch.Tensor], device):
    images = torch.stack(image_list)
    return images.to(device)


def pad_img(image: torch.Tensor) -> torch.Tensor:
    """
    Add padding to the image to resize it to be 224x224
    """
    padding = [0, RESNET_INPUT_SIZE -
               image.size()[2], 0, RESNET_INPUT_SIZE - image.size()[1], 0, 0]
    resize_img = F.pad(image, padding)
    return resize_img


def clean_batch(subject_images: List[torch.Tensor], verb_images: List[torch.Tensor], object_images: List[torch.Tensor], subject_labels: List[torch.Tensor], verb_labels: List[torch.Tensor], object_labels: List[torch.Tensor]):
    '''
    Mutatively remove all of the none types from the batches and add padding to 
    fit 224x224 image size for the resent18 backbone
    '''
    assert len(subject_images) == len(verb_images) == len(object_images) == len(
        subject_labels) == len(verb_labels) == len(object_labels)
    i = 0
    while i < len(subject_images):
        if subject_images[i] is None or verb_images[i] is None or object_images[i] is None:
            subject_images.pop(i)
            verb_images.pop(i)
            object_images.pop(i)
            subject_labels.pop(i)
            verb_labels.pop(i)
            object_labels.pop(i)
        else:
            subject_images[i] = pad_img(subject_images[i])
            verb_images[i] = pad_img(verb_images[i])
            object_images[i] = pad_img(object_images[i])
            i += 1
