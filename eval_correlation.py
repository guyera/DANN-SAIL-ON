import copy

import matplotlib.pyplot as plt
import sklearn.metrics
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34
from data.svodataset import SVODataset
#from featurizer import Featurizer
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from typing import List, Optional, Tuple
import os
import shutil
import argparse
#import wandb
import numpy as np
from utils import *

NUM_SUBJECTS = 5
NUM_VERBS = 8
NUM_OBJECTS = 12


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("GPU detected")
else:
    print("No GPU dectected, defaulting to CPU")

def state_dict(module):
    return {k: v.cpu() for k, v in module.state_dict().items()}

def custom_collate(batch):
    subject_images = []
    subject_labels = []
    verb_images = []
    verb_labels = []
    object_images = []
    object_labels = []
    spatial_encodings = []
    for subject_image, verb_image, object_image, spatial_encoding, subject_label, verb_label, object_label in batch:
        subject_images.append(subject_image)
        verb_images.append(verb_image)
        object_images.append(object_image)
        spatial_encodings.append(spatial_encoding)
        subject_labels.append(subject_label)
        verb_labels.append(verb_label)
        object_labels.append(object_label)
    
    return subject_images, verb_images, object_images, None, subject_labels, verb_labels, object_labels

def eval_correlation(args: argparse.Namespace, data_loader, subject_feature_extractor, subject_classifier, verb_feature_extractor, verb_classifier, object_feature_extractor, object_classifier):
    subject_feature_extractor.eval()
    subject_classifier.eval()
    verb_feature_extractor.eval()
    verb_classifier.eval()
    object_feature_extractor.eval()
    object_classifier.eval()
    
    all_subject_scores = []
    all_subject_labels = []
    all_verb_scores = []
    all_verb_labels = []
    all_object_scores = []
    all_object_labels = []

    correct = 0
    total = 0
    
    with torch.no_grad():
        for subject_images, verb_images, object_images, spatial_encodings, subject_labels, verb_labels, object_labels in data_loader:
            present_subject_indices = [idx for idx, label in enumerate(subject_labels) if label is not None]
            present_subject_images = torch.stack([subject_images[idx] for idx in present_subject_indices], dim = 0).to(device) if len(present_subject_indices) > 0 else None
            present_subject_features = subject_feature_extractor(present_subject_images) if len(present_subject_indices) > 0 else None
            present_subject_logits = subject_classifier(present_subject_features) if len(present_subject_indices) > 0 else None
            present_subject_filtered_logits = present_subject_logits[:, 1:] if len(present_subject_indices) > 0 else None
            present_subject_max_logits = torch.max(present_subject_filtered_logits, dim = 1)[0] if len(present_subject_indices) > 0 else None
            present_subject_scores = -present_subject_max_logits if len(present_subject_indices) > 0 else None
            present_subject_itr = 0
            for subject_label in subject_labels:
                if subject_label is None:
                    all_subject_labels.append(None)
                    all_subject_scores.append(None)
                else:
                    all_subject_labels.append(subject_label)
                    all_subject_scores.append(present_subject_scores[present_subject_itr])
                    present_subject_itr += 1

            present_verb_indices = [idx for idx, label in enumerate(verb_labels) if label is not None]
            present_verb_images = torch.stack([verb_images[idx] for idx in present_verb_indices], dim = 0).to(device) if len(present_verb_indices) > 0 else None
            present_verb_features = verb_feature_extractor(present_verb_images) if len(present_verb_indices) > 0 else None
            if spatial_encodings is not None:
                spatial_encodings = spatial_encodings.to(device)
                spatial_encodings = torch.flatten(spatial_encodings, start_dim = 1)
                present_verb_features = torch.cat((spatial_encodings, present_verb_features), dim = 1)
            present_verb_logits = verb_classifier(present_verb_features) if len(present_verb_indices) > 0 else None
            present_verb_filtered_logits = present_verb_logits[:, 1:] if len(present_verb_indices) > 0 else None
            present_verb_max_logits = torch.max(present_verb_filtered_logits, dim = 1)[0] if len(present_verb_indices) > 0 else None
            present_verb_scores = -present_verb_max_logits if len(present_verb_indices) > 0 else None
            present_verb_itr = 0
            for verb_label in verb_labels:
                if verb_label is None:
                    all_verb_labels.append(None)
                    all_verb_scores.append(None)
                else:
                    all_verb_labels.append(verb_label)
                    all_verb_scores.append(present_verb_scores[present_verb_itr])
                    present_verb_itr += 1

            present_object_indices = [idx for idx, label in enumerate(object_labels) if label is not None]
            present_object_images = torch.stack([object_images[idx] for idx in present_object_indices], dim = 0).to(device) if len(present_object_indices) > 0 else None
            present_object_features = object_feature_extractor(present_object_images) if len(present_object_indices) > 0 else None
            present_object_logits = object_classifier(present_object_features) if len(present_object_indices) > 0 else None
            present_object_filtered_logits = present_object_logits[:, 1:] if len(present_object_indices) > 0 else None
            present_object_max_logits = torch.max(present_object_filtered_logits, dim = 1)[0] if len(present_object_indices) > 0 else None
            present_object_scores = -present_object_max_logits if len(present_object_indices) > 0 else None
            present_object_itr = 0
            for object_label in object_labels:
                if object_label is None:
                    all_object_labels.append(None)
                    all_object_scores.append(None)
                else:
                    all_object_labels.append(object_label)
                    all_object_scores.append(present_object_scores[present_object_itr])
                    present_object_itr += 1
        
        return all_subject_scores, all_subject_labels, all_verb_scores, all_verb_labels, all_object_scores, all_object_labels

class AdaptivePad:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, x):
        horizontal_padding = self.size - x.shape[2]
        left_padding = horizontal_padding // 2
        right_padding = horizontal_padding - left_padding
        
        vertical_padding = self.size - x.shape[1]
        top_padding = vertical_padding // 2
        bottom_padding = vertical_padding - top_padding
        
        return torchvision.transforms.functional.pad(x, [left_padding, top_padding, right_padding, bottom_padding])

class NullableWrapperTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        if x is not None:
            x = self.transform(x)
        return x

def main(args):
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = args.cudnn_benchmark

    transform = NullableWrapperTransform(torchvision.transforms.Compose([torchvision.transforms.Resize(223, max_size = 224), AdaptivePad(224), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    val_dataset = SVODataset(
        name='Custom',
        data_root='Custom',
        csv_path='Custom/annotations/dataset_v4_val.csv',
        transform = transform
    )

    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )
    
    subject_feature_extractor = resnet18(pretrained=False)
    subject_feature_extractor.fc = torch.nn.Linear(subject_feature_extractor.fc.weight.shape[1], args.bottleneck_dim)
    subject_classifier = torch.nn.Linear(args.bottleneck_dim, NUM_SUBJECTS)

    subject_state_dicts = torch.load(args.subject_classifier_load_file)
    subject_feature_extractor.load_state_dict(subject_state_dicts['feature_extractor'])
    subject_classifier.load_state_dict(subject_state_dicts['classifier'])
    
    subject_feature_extractor = subject_feature_extractor.to(device)
    subject_classifier = subject_classifier.to(device)
    
    verb_feature_extractor = resnet18(pretrained=False)
    verb_feature_extractor.fc = torch.nn.Linear(verb_feature_extractor.fc.weight.shape[1], args.bottleneck_dim)
    #verb_classifier = torch.nn.Linear(args.bottleneck_dim + args.spatial_encoding_dim, NUM_VERBS)
    verb_classifier = torch.nn.Linear(args.bottleneck_dim, NUM_VERBS)

    verb_state_dicts = torch.load(args.verb_classifier_load_file)
    verb_feature_extractor.load_state_dict(verb_state_dicts['feature_extractor'])
    verb_classifier.load_state_dict(verb_state_dicts['classifier'])
    
    verb_feature_extractor = verb_feature_extractor.to(device)
    verb_classifier = verb_classifier.to(device)
    
    object_feature_extractor = resnet18(pretrained=False)
    object_feature_extractor.fc = torch.nn.Linear(object_feature_extractor.fc.weight.shape[1], args.bottleneck_dim)
    object_classifier = torch.nn.Linear(args.bottleneck_dim, NUM_OBJECTS)

    object_state_dicts = torch.load(args.object_classifier_load_file)
    object_feature_extractor.load_state_dict(object_state_dicts['feature_extractor'])
    object_classifier.load_state_dict(object_state_dicts['classifier'])
    
    object_feature_extractor = object_feature_extractor.to(device)
    object_classifier = object_classifier.to(device)
    
    subject_novelty_scores, subject_labels, verb_novelty_scores, verb_labels, object_novelty_scores, object_labels = eval_correlation(args, data_loader, subject_feature_extractor, subject_classifier, verb_feature_extractor, verb_classifier, object_feature_extractor, object_classifier)
    
    sv_filtered_subject_scores = []
    so_filtered_subject_scores = []

    sv_filtered_verb_scores = []
    vo_filtered_verb_scores = []

    so_filtered_object_scores = []
    vo_filtered_object_scores = []
    for idx in range(len(subject_novelty_scores)):
        subject_score = subject_novelty_scores[idx]
        verb_score = verb_novelty_scores[idx]
        object_score = object_novelty_scores[idx]
        
        if subject_score is not None:
            if verb_score is not None:
                sv_filtered_subject_scores.append(subject_score)
            if object_score is not None:
                so_filtered_subject_scores.append(subject_score)
        
        if verb_score is not None:
            if subject_score is not None:
                sv_filtered_verb_scores.append(verb_score)
            if object_score is not None:
                vo_filtered_verb_scores.append(verb_score)
        
        if object_score is not None:
            if subject_score is not None:
                so_filtered_object_scores.append(object_score)
            if verb_score is not None:
                vo_filtered_object_scores.append(object_score)
        
    sv_filtered_subject_scores = torch.stack(sv_filtered_subject_scores, dim = 0)
    so_filtered_subject_scores = torch.stack(so_filtered_subject_scores, dim = 0)

    sv_filtered_verb_scores = torch.stack(sv_filtered_verb_scores, dim = 0)
    vo_filtered_verb_scores = torch.stack(vo_filtered_verb_scores, dim = 0)

    so_filtered_object_scores = torch.stack(so_filtered_object_scores, dim = 0)
    vo_filtered_object_scores = torch.stack(vo_filtered_object_scores, dim = 0)

    if not os.path.exists(args.figure_dir):
        os.makedirs(args.figure_dir)

# Verb vs. subject scatter plot
    fig, ax = plt.subplots()
    ax.set_title('Verb novelty scores vs subject novelty scores')
    ax.set_xlabel('Subject novelty scores')
    ax.set_ylabel('Verb novelty scores')
    ax.scatter(sv_filtered_subject_scores.detach().cpu().numpy(), sv_filtered_verb_scores.detach().cpu().numpy())
    a = torch.stack((sv_filtered_subject_scores, torch.ones_like(sv_filtered_subject_scores)), dim = 1)
    y = sv_filtered_verb_scores
    x = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(a.T, a)), a.T), y)
    y_hat = torch.matmul(a, x)
    ax.plot(sv_filtered_subject_scores.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), color = 'black')
    r = torch.corrcoef(torch.stack((sv_filtered_subject_scores, sv_filtered_verb_scores), dim = 0))[0, 1]
    text = f'r = {float(r):.2f}'
    props = dict(boxstyle = 'round', facecolor = 'tab:orange', alpha = 0.85)
    ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize = 14, verticalalignment = 'bottom', horizontalalignment = 'right', bbox = props)
    fig.savefig(os.path.join(args.figure_dir, 'subject_verb_scatter.jpg'))
    plt.close(fig)

# Object vs. subject scatter plot
    fig, ax = plt.subplots()
    ax.set_title('Object novelty scores vs subject novelty scores')
    ax.set_xlabel('Subject novelty scores')
    ax.set_ylabel('Object novelty scores')
    ax.scatter(so_filtered_subject_scores.detach().cpu().numpy(), so_filtered_object_scores.detach().cpu().numpy())
    a = torch.stack((so_filtered_subject_scores, torch.ones_like(so_filtered_subject_scores)), dim = 1)
    y = so_filtered_object_scores
    x = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(a.T, a)), a.T), y)
    y_hat = torch.matmul(a, x)
    ax.plot(so_filtered_subject_scores.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), color = 'black')
    r = torch.corrcoef(torch.stack((so_filtered_subject_scores, so_filtered_object_scores), dim = 0))[0, 1]
    text = f'r = {float(r):.2f}'
    props = dict(boxstyle = 'round', facecolor = 'tab:orange', alpha = 0.85)
    ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize = 14, verticalalignment = 'bottom', horizontalalignment = 'right', bbox = props)
    fig.savefig(os.path.join(args.figure_dir, 'subject_object_scatter.jpg'))
    plt.close(fig)

# Object vs. verb scatter plot
    fig, ax = plt.subplots()
    ax.set_title('Object novelty scores vs verb novelty scores')
    ax.set_xlabel('Verb novelty scores')
    ax.set_ylabel('Object novelty scores')
    ax.scatter(vo_filtered_verb_scores.detach().cpu().numpy(), vo_filtered_object_scores.detach().cpu().numpy())
    a = torch.stack((vo_filtered_verb_scores, torch.ones_like(vo_filtered_verb_scores)), dim = 1)
    y = vo_filtered_object_scores
    x = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(a.T, a)), a.T), y)
    y_hat = torch.matmul(a, x)
    ax.plot(vo_filtered_verb_scores.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), color = 'black')
    r = torch.corrcoef(torch.stack((vo_filtered_verb_scores, vo_filtered_object_scores), dim = 0))[0, 1]
    text = f'r = {float(r):.2f}'
    props = dict(boxstyle = 'round', facecolor = 'tab:orange', alpha = 0.85)
    ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize = 14, verticalalignment = 'bottom', horizontalalignment = 'right', bbox = props)
    fig.savefig(os.path.join(args.figure_dir, 'verb_object_scatter.jpg'))
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Application of DANN on the Sail-On dataset')
    parser.add_argument('--bottleneck-dim',
                        default=256,
                        type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--spatial-encoding-dim',
                        default=72,
                        type=int,
                        help='Dimension of flattened spatial encodings')
    parser.add_argument('-b',
                        '--batch-size',
                        default=64,
                        type=int,
                        metavar='N',
                        help='mini-batch size used for evaluating (default: 32)')
    parser.add_argument(\
        '--subject-classifier-load-file',
        type = str,
        required = True
    )
    parser.add_argument(\
        '--verb-classifier-load-file',
        type = str,
        required = True
    )
    parser.add_argument(\
        '--object-classifier-load-file',
        type = str,
        required = True
    )
    parser.add_argument(\
        '--figure-dir',
        type = str,
        required = True
    )
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cudnn-benchmark',
                        default=True,
                        type=bool,
                        help='flag for torch.cudnn_benchmark')
    args = parser.parse_args()

    main(args)
