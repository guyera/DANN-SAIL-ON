import copy

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
    images = []
    spatial_encodings = []
    novelty_labels = []
    for image, spatial_encoding, novelty_label in batch:
        images.append(image)
        spatial_encodings.append(spatial_encoding)
        novelty_labels.append(novelty_label)
    
    images = torch.stack(images, dim = 0)
    novelty_labels = torch.stack(novelty_labels, dim = 0)
    spatial_encodings = torch.stack(spatial_encodings, dim = 0) if spatial_encodings[0] is not None else None

    return images, spatial_encodings, novelty_labels

def eval_auc(args: argparse.Namespace, anomaly_detection_loader: DataLoader, feature_extractor, classifier):
    feature_extractor.eval()
    classifier.eval()
    
    novelty_scores = []
    novelty_labels = []

    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_images, batch_spatial_encodings, batch_labels in anomaly_detection_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_novelty_indices = batch_labels == 0
            batch_novelty_labels = batch_novelty_indices.to(torch.long)
            
            features = feature_extractor(batch_images)

            # If we're evaluating the verb classifier, then make sure the verb
            # classifier has access to the spatial encodings.
            if batch_spatial_encodings is not None:
                batch_spatial_encodings = batch_spatial_encodings.to(device)
                batch_spatial_encodings = torch.flatten(batch_spatial_encodings, start_dim = 1)
                features = torch.cat((batch_spatial_encodings, features), dim = 1)
            
            logits = classifier(features)
            
            known_example_logits = logits[~batch_novelty_indices]
            known_example_predictions = torch.argmax(known_example_logits, dim = 1)
            batch_known_example_labels = batch_labels[~batch_novelty_indices]
            batch_correct = (known_example_predictions == batch_known_example_labels).to(torch.int).sum()
            correct += batch_correct.detach().cpu().item()
            total += known_example_logits.shape[0]
            
            logits = logits[:, 1:] # Remove class-0 label
            max_logits, _ = torch.max(logits, dim = 1)
            batch_novelty_scores = -max_logits
            
            novelty_scores.append(batch_novelty_scores)
            novelty_labels.append(batch_novelty_labels)
        
        novelty_scores = torch.cat(novelty_scores, dim = 0)
        novelty_labels = torch.cat(novelty_labels, dim = 0)
        
        accuracy = float(correct) / total
        auc = sklearn.metrics.roc_auc_score(novelty_labels.detach().cpu().numpy(), novelty_scores.detach().cpu().numpy())
        
        return auc, accuracy

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

class SubjectNoveltyDetectionSVODataset(torch.utils.data.Dataset):
    def __init__(self, svo_dataset):
        super().__init__()
        self.svo_dataset = svo_dataset
    
    def __len__(self):
        return len(self.svo_dataset)

    def __getitem__(self, idx):
        image, _, _, _, subject_label, _, _ = self.svo_dataset[idx]
        return image, None, subject_label

class VerbNoveltyDetectionSVODataset(torch.utils.data.Dataset):
    def __init__(self, svo_dataset):
        super().__init__()
        self.svo_dataset = svo_dataset
    
    def __len__(self):
        return len(self.svo_dataset)

    def __getitem__(self, idx):
        _, image, _, spatial_encodings, _, verb_label, _ = self.svo_dataset[idx]
        #return image, spatial_encodings, verb_label
        return image, None, verb_label

class ObjectNoveltyDetectionSVODataset(torch.utils.data.Dataset):
    def __init__(self, svo_dataset):
        super().__init__()
        self.svo_dataset = svo_dataset
    
    def __len__(self):
        return len(self.svo_dataset)

    def __getitem__(self, idx):
        _, _, image, _, _, _, object_label = self.svo_dataset[idx]
        return image, None, object_label
    
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

    subject_indices = []
    verb_indices = []
    object_indices = []
    num_verb_novelty = 0
    for idx, (subject_image, verb_image, object_image, spatial_encodings, subject_label, verb_label, object_label) in enumerate(val_dataset):
        # Find non-null examples
        if subject_label is not None:
            subject_indices.append(idx)
        if verb_label is not None:
            verb_indices.append(idx)
            if verb_label == 0:
                num_verb_novelty += 1
        if object_label is not None:
            object_indices.append(idx)
    
    print(f'Number of verb novelties: {num_verb_novelty}')
    subject_dataset = SubjectNoveltyDetectionSVODataset(torch.utils.data.Subset(val_dataset, subject_indices))
    verb_dataset = VerbNoveltyDetectionSVODataset(torch.utils.data.Subset(val_dataset, verb_indices))
    object_dataset = ObjectNoveltyDetectionSVODataset(torch.utils.data.Subset(val_dataset, object_indices))

    subject_data_loader = torch.utils.data.DataLoader(
        subject_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )
    verb_data_loader = torch.utils.data.DataLoader(
        verb_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )
    object_data_loader = torch.utils.data.DataLoader(
        object_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )
    
    print('#################################')
    print('# Evaluating subject classifier #')
    print('#################################')
    subject_feature_extractor = resnet18(pretrained=False)
    subject_feature_extractor.fc = torch.nn.Linear(subject_feature_extractor.fc.weight.shape[1], args.bottleneck_dim)
    subject_classifier = torch.nn.Linear(args.bottleneck_dim, NUM_SUBJECTS)

    subject_state_dicts = torch.load(args.subject_classifier_load_file)
    subject_feature_extractor.load_state_dict(subject_state_dicts['feature_extractor'])
    subject_classifier.load_state_dict(subject_state_dicts['classifier'])
    
    subject_feature_extractor = subject_feature_extractor.to(device)
    subject_classifier = subject_classifier.to(device)
    
    subject_auc, subject_accuracy = eval_auc(args, subject_data_loader, subject_feature_extractor, subject_classifier)
    
    print(f'Subject AUC: {subject_auc}')
    print(f'Subject accuracy: {subject_accuracy}')

    print('##################################')
    print('#   Evaluating verb classifier   #')
    print('##################################')
    verb_feature_extractor = resnet18(pretrained=False)
    verb_feature_extractor.fc = torch.nn.Linear(verb_feature_extractor.fc.weight.shape[1], args.bottleneck_dim)
    #verb_classifier = torch.nn.Linear(args.bottleneck_dim + args.spatial_encoding_dim, NUM_VERBS)
    verb_classifier = torch.nn.Linear(args.bottleneck_dim, NUM_VERBS)

    verb_state_dicts = torch.load(args.verb_classifier_load_file)
    verb_feature_extractor.load_state_dict(verb_state_dicts['feature_extractor'])
    verb_classifier.load_state_dict(verb_state_dicts['classifier'])
    
    verb_feature_extractor = verb_feature_extractor.to(device)
    verb_classifier = verb_classifier.to(device)
    
    verb_auc, verb_accuracy = eval_auc(args, verb_data_loader, verb_feature_extractor, verb_classifier)

    print(f'Verb AUC: {verb_auc}')
    print(f'Verb accuracy: {verb_accuracy}')

    print('##################################')
    print('#  Evaluating object classifier  #')
    print('##################################')
    object_feature_extractor = resnet18(pretrained=False)
    object_feature_extractor.fc = torch.nn.Linear(object_feature_extractor.fc.weight.shape[1], args.bottleneck_dim)
    object_classifier = torch.nn.Linear(args.bottleneck_dim, NUM_OBJECTS)

    object_state_dicts = torch.load(args.object_classifier_load_file)
    object_feature_extractor.load_state_dict(object_state_dicts['feature_extractor'])
    object_classifier.load_state_dict(object_state_dicts['classifier'])
    
    object_feature_extractor = object_feature_extractor.to(device)
    object_classifier = object_classifier.to(device)
    
    object_auc, object_accuracy = eval_auc(args, object_data_loader, object_feature_extractor, object_classifier)

    print(f'Object AUC: {object_auc}')
    print(f'Object accuracy: {object_accuracy}')

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
