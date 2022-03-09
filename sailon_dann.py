import copy

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34
from grl import GradientReverseLayer, WarmStartGradientReverseLayer
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
    classifier_labels = []
    adversary_1_labels = []
    adversary_2_labels = []
    for image, spatial_encoding, classifier_label, adversary_1_label, adversary_2_label in batch:
        images.append(image)
        spatial_encodings.append(spatial_encoding)
        classifier_labels.append(classifier_label)
        adversary_1_labels.append(adversary_1_label)
        adversary_2_labels.append(adversary_2_label)

    return images, spatial_encodings, classifier_labels, adversary_1_labels, adversary_2_labels

def train(args: argparse.Namespace, train_loader: DataLoader, feature_extractor, classifier, adversary_1, adversary_2, classifier_optimizer: SGD, adversary_optimizer: SGD):
    feature_extractor.train()
    classifier.train()
    adversary_1.train()
    adversary_2.train()

    total_classifier_examples = 0
    total_adversary_1_examples = 0
    total_adversary_2_examples = 0
    
    total_loss = 0.0
    total_classifier_loss = 0.0
    total_adversary_1_loss = 0.0
    total_adversary_2_loss = 0.0

    total_classifier_accuracy = 0.0
    total_adversary_1_accuracy = 0.0
    total_adversary_2_accuracy = 0.0
    
    train_log = {}
    
    step_adversary = False
    step_type_counter = 0.0

    for images, spatial_encodings, classifier_labels, adversary_1_labels, adversary_2_labels in train_loader:
        step_adversary = args.train_adversaries and step_type_counter < 0.0
        
        images = torch.stack(images, dim = 0).to(device)
        classifier_labels = torch.stack(classifier_labels, dim = 0).to(device)
        
        adversary_1_present = [idx for idx, label in enumerate(adversary_1_labels) if label is not None]
        adversary_2_present = [idx for idx, label in enumerate(adversary_2_labels) if label is not None]
        adversary_1_present_labels = torch.stack([adversary_1_labels[i] for i in adversary_1_present], dim = 0).to(device)
        adversary_2_present_labels = torch.stack([adversary_2_labels[i] for i in adversary_2_present], dim = 0).to(device)
    
        features = feature_extractor(images)
        adversary_1_features = features[adversary_1_present]
        adversary_2_features = features[adversary_2_present]
            
        # If we're training the verb classifier, then make sure the verb
        # classifier has access to the spatial encodings. But the subject and
        # object adversaries shouldn't have access to those encodings.
        if spatial_encodings[0] is not None:
            spatial_encodings = torch.stack(spatial_encodings, dim = 0).to(device)
            spatial_encodings = torch.flatten(spatial_encodings, start_dim = 1)
            features = torch.cat((spatial_encodings, features), dim = 1)
        
        classifier_preds = classifier(features)
        adversary_1_preds = adversary_1(adversary_1_features)
        adversary_2_preds = adversary_2(adversary_2_features)
        
        classifier_loss = F.cross_entropy(classifier_preds, classifier_labels)
        adversary_1_loss = F.cross_entropy(adversary_1_preds, adversary_1_present_labels)
        adversary_2_loss = F.cross_entropy(adversary_2_preds, adversary_2_present_labels)
        
        if step_adversary:
            loss = adversary_1_loss + adversary_2_loss
            adversary_optimizer.zero_grad()
            loss.backward()
            adversary_optimizer.step()
        else:
            if not args.train_adversaries:
                loss = classifier_loss
            else:
                loss = classifier_loss - args.adversary_trade_off * (adversary_1_loss + adversary_2_loss)
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

        # Logging losses and accuracies
        total_classifier_examples += features.shape[0]
        total_adversary_1_examples += adversary_1_features.shape[0]
        total_adversary_2_examples += adversary_2_features.shape[0]
        
        total_loss += loss.item() * features.shape[0]
        total_classifier_loss += classifier_loss.item() * features.shape[0]
        total_adversary_1_loss += adversary_1_loss.item() * adversary_1_features.shape[0]
        total_adversary_2_loss += adversary_2_loss.item() * adversary_2_features.shape[0]
        
        total_classifier_accuracy += accuracy(classifier_preds, classifier_labels) * features.shape[0]
        total_adversary_1_accuracy += accuracy(adversary_1_preds, adversary_1_present_labels) * adversary_1_features.shape[0]
        total_adversary_2_accuracy += accuracy(adversary_2_preds, adversary_2_present_labels) * adversary_2_features.shape[0]
        
        if not step_adversary:
            step_type_counter -= 1.0
        else:
            step_type_counter += 1.0 / args.adversary_step_ratio

    mean_loss = total_loss / total_classifier_examples
    mean_classifier_loss = total_classifier_loss / total_classifier_examples
    mean_adversary_1_loss = total_adversary_1_loss / total_adversary_1_examples
    mean_adversary_2_loss = total_adversary_2_loss / total_adversary_2_examples
    
    mean_classifier_accuracy = total_classifier_accuracy / total_classifier_examples
    mean_adversary_1_accuracy = total_adversary_1_accuracy / total_adversary_1_examples
    mean_adversary_2_accuracy = total_adversary_2_accuracy / total_adversary_2_examples
    
    train_log.update({
        "Total Loss (Training Set)": mean_loss,
        "Classifier Loss (Training Set)": mean_classifier_loss,
        "Adversary 1 Loss (Training Set)": mean_adversary_1_loss,
        "Adversary 2 Loss (Training Set)": mean_adversary_2_loss,
        "Classifier Accuracy (Training Set)": mean_classifier_accuracy,
        "Adversary 1 Accuracy (Training Set)": mean_adversary_1_accuracy,
        "Adversary 2 Accuracy (Training Set)": mean_adversary_2_accuracy
    })

    return train_log

def finetune_adversaries(args: argparse.Namespace, train_loader: DataLoader, feature_extractor, adversary_1, adversary_2, adversary_optimizer: SGD):
    feature_extractor.eval()
    adversary_1.eval()
    adversary_2.eval()
    
    all_adversary_1_labels = []
    all_adversary_2_labels = []
    all_adversary_1_features = []
    all_adversary_2_features = []
    
    with torch.no_grad():
        for images, _, _, adversary_1_labels, adversary_2_labels in train_loader:
            images = torch.stack(images, dim = 0).to(device)
            
            adversary_1_present = [idx for idx, label in enumerate(adversary_1_labels) if label is not None]
            adversary_2_present = [idx for idx, label in enumerate(adversary_2_labels) if label is not None]
            adversary_1_present_labels = torch.stack([adversary_1_labels[i] for i in adversary_1_present], dim = 0).to(device)
            adversary_2_present_labels = torch.stack([adversary_2_labels[i] for i in adversary_2_present], dim = 0).to(device)
            
            features = feature_extractor(images)
            adversary_1_features = features[adversary_1_present]
            adversary_2_features = features[adversary_2_present]
            
            all_adversary_1_labels.append(adversary_1_present_labels)
            all_adversary_2_labels.append(adversary_2_present_labels)
            all_adversary_1_features.append(adversary_1_features)
            all_adversary_2_features.append(adversary_2_features)
            
    all_adversary_1_labels = torch.cat(all_adversary_1_labels, dim = 0)
    all_adversary_2_labels = torch.cat(all_adversary_2_labels, dim = 0)
    all_adversary_1_features = torch.cat(all_adversary_1_features, dim = 0)
    all_adversary_2_features = torch.cat(all_adversary_2_features, dim = 0)
    
    for epoch in range(args.adversary_finetuning_epochs):
        adversary_1_preds = adversary_1(all_adversary_1_features)
        adversary_2_preds = adversary_2(all_adversary_2_features)
        
        adversary_1_loss = F.cross_entropy(adversary_1_preds, all_adversary_1_labels)
        adversary_2_loss = F.cross_entropy(adversary_2_preds, all_adversary_2_labels)
        
        loss = adversary_1_loss + adversary_2_loss
        adversary_optimizer.zero_grad()
        loss.backward()
        adversary_optimizer.step()

def validate(args: argparse.Namespace, val_loader: DataLoader, feature_extractor, classifier, adversary_1, adversary_2, dataset_type: Optional[str] = 'Validation'):
    feature_extractor.eval()
    classifier.eval()
    adversary_1.eval()
    adversary_2.eval()
    
    total_classifier_examples = 0
    total_adversary_1_examples = 0
    total_adversary_2_examples = 0
    
    total_loss = 0.0
    total_classifier_loss = 0.0
    total_adversary_1_loss = 0.0
    total_adversary_2_loss = 0.0

    total_classifier_accuracy = 0.0
    total_adversary_1_accuracy = 0.0
    total_adversary_2_accuracy = 0.0

    val_log = {}
    
    with torch.no_grad():
        for images, spatial_encodings, classifier_labels, adversary_1_labels, adversary_2_labels in val_loader:
            images = torch.stack(images, dim = 0).to(device)
            classifier_labels = torch.stack(classifier_labels, dim = 0).to(device)

            adversary_1_present = [idx for idx, label in enumerate(adversary_1_labels) if label is not None]
            adversary_2_present = [idx for idx, label in enumerate(adversary_2_labels) if label is not None]
            adversary_1_present_labels = torch.stack([adversary_1_labels[i] for i in adversary_1_present], dim = 0).to(device)
            adversary_2_present_labels = torch.stack([adversary_2_labels[i] for i in adversary_2_present], dim = 0).to(device)
            
            features = feature_extractor(images)
            adversary_1_present_features = features[adversary_1_present]
            adversary_2_present_features = features[adversary_2_present]

            # If we're training the verb classifier, then make sure the verb
            # classifier has access to the spatial encodings. But the subject and
            # object adversaries shouldn't have access to those encodings.
            if spatial_encodings[0] is not None:
                spatial_encodings = torch.stack(spatial_encodings, dim = 0).to(device)
                spatial_encodings = torch.flatten(spatial_encodings, start_dim = 1)
                features = torch.cat((spatial_encodings, features), dim = 1)
            
            classifier_preds = classifier(features)
            adversary_1_preds = adversary_1(adversary_1_present_features)
            adversary_2_preds = adversary_2(adversary_2_present_features)
            
            classifier_loss = F.cross_entropy(classifier_preds, classifier_labels)
            adversary_1_loss = F.cross_entropy(adversary_1_preds, adversary_1_present_labels)
            adversary_2_loss = F.cross_entropy(adversary_2_preds, adversary_2_present_labels)
            
            loss = classifier_loss - adversary_1_loss - adversary_2_loss
            
            total_classifier_examples += features.shape[0]
            total_adversary_1_examples += adversary_1_present_features.shape[0]
            total_adversary_2_examples += adversary_2_present_features.shape[0]
            
            total_loss += loss.item() * features.shape[0]
            total_classifier_loss += classifier_loss.item() * features.shape[0]
            total_adversary_1_loss += adversary_1_loss.item() * adversary_1_present_features.shape[0]
            total_adversary_2_loss += adversary_2_loss.item() * adversary_2_present_features.shape[0]
            
            total_classifier_accuracy += accuracy(classifier_preds, classifier_labels) * features.shape[0]
            total_adversary_1_accuracy += accuracy(adversary_1_preds, adversary_1_present_labels) * adversary_1_present_features.shape[0]
            total_adversary_2_accuracy += accuracy(adversary_2_preds, adversary_2_present_labels) * adversary_2_present_features.shape[0]
            
    mean_loss = total_loss / total_classifier_examples
    mean_classifier_loss = total_classifier_loss / total_classifier_examples
    mean_adversary_1_loss = total_adversary_1_loss / total_adversary_1_examples
    mean_adversary_2_loss = total_adversary_2_loss / total_adversary_2_examples

    mean_classifier_accuracy = total_classifier_accuracy / total_classifier_examples
    mean_adversary_1_accuracy = total_adversary_1_accuracy / total_adversary_1_examples
    mean_adversary_2_accuracy = total_adversary_2_accuracy / total_adversary_2_examples

    val_log.update({
        f"Total Loss ({dataset_type} Set)": mean_loss,
        f"Classifier Loss ({dataset_type} Set)": mean_classifier_loss,
        f"Adversary 1 Loss ({dataset_type} Set)": mean_adversary_1_loss,
        f"Adversary 2 Loss ({dataset_type} Set)": mean_adversary_2_loss,
        f"Classifier Accuracy ({dataset_type} Set)": mean_classifier_accuracy,
        f"Adversary 1 Accuracy ({dataset_type} Set)": mean_adversary_1_accuracy,
        f"Adversary 2 Accuracy ({dataset_type} Set)": mean_adversary_2_accuracy
    })
    
    return val_log, mean_loss, mean_classifier_accuracy

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

class SubjectImageSVODataset(torch.utils.data.Dataset):
    def __init__(self, svo_dataset):
        super().__init__()
        self.svo_dataset = svo_dataset
    
    def __len__(self):
        return len(self.svo_dataset)

    def __getitem__(self, idx):
        images, _, _, _, subject_labels, verb_labels, object_labels = self.svo_dataset[idx]
        return images, None, subject_labels, verb_labels, object_labels

class VerbImageSVODataset(torch.utils.data.Dataset):
    def __init__(self, svo_dataset):
        super().__init__()
        self.svo_dataset = svo_dataset
    
    def __len__(self):
        return len(self.svo_dataset)

    def __getitem__(self, idx):
        _, images, _, spatial_encodings, subject_labels, verb_labels, object_labels = self.svo_dataset[idx]
        return images, spatial_encodings, verb_labels, subject_labels, object_labels

class ObjectImageSVODataset(torch.utils.data.Dataset):
    def __init__(self, svo_dataset):
        super().__init__()
        self.svo_dataset = svo_dataset
    
    def __len__(self):
        return len(self.svo_dataset)
    
    def __getitem__(self, idx):
        _, _, images, _, subject_labels, verb_labels, object_labels = self.svo_dataset[idx]
        return images, None, object_labels, subject_labels, verb_labels

def run(args, dataset, classifier, adversary_1, adversary_2, save_subdir):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )

    feature_extractor = resnet18(pretrained=True)
    #feature_extractor = resnet34(pretrained=True)
    feature_extractor.fc = torch.nn.Linear(feature_extractor.fc.weight.shape[1], args.bottleneck_dim)
    feature_extractor = feature_extractor.to(device)

    classifier_optimizer = SGD(list(feature_extractor.parameters()) + list(classifier.parameters()),
                         args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay,
                         nesterov=True)
    
    adversary_optimizer = SGD(list(adversary_1.parameters()) + list(adversary_2.parameters()),
                   args.adversary_lr,
                   momentum=args.momentum,
                   weight_decay=args.weight_decay,
                   nesterov=True)

    best_loss = None
    best_loss_state_dicts = None
    best_accuracy = None
    best_accuracy_state_dicts = None

    print("\n######## STARTING TRAINING LOOP #########")
    for epoch in range(args.epochs):
        print(f"\n######## START TRAINING FOR EPOCH {epoch} #########")
        train_log = train(args, train_loader, feature_extractor,
                          classifier, adversary_1, adversary_2, classifier_optimizer, adversary_optimizer)
        print(f"Epoch {epoch} Training Results")
        #wandb.log(train_log)
        print_log(train_log)
        print(f"######## END TRAINING FOR EPOCH {epoch} #########\n")

        print(f"\n######## START VALIDATION FOR EPOCH {epoch} #########")
        
        val_log, val_loss, val_accuracy = validate(
            args, val_loader, feature_extractor, classifier, adversary_1, adversary_2)
        print(f"Epoch {epoch} Validation Results")
        print_log(val_log)

        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            best_loss_state_dicts = copy.deepcopy({
                'feature_extractor': feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'adversary_1': adversary_1.state_dict(),
                'adversary_2': adversary_2.state_dict()
            })
        if best_accuracy is None or val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_accuracy_state_dicts = copy.deepcopy({
                'feature_extractor': feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'adversary_1': adversary_1.state_dict(),
                'adversary_2': adversary_2.state_dict()
            })
            
        print(f"######## END VALIDATION FOR EPOCH {epoch} #########\n")
    
    print("######## ENDING TRAINING LOOP #########\n")
    
    print("Loading best model based on validation loss...", end = "")
    feature_extractor.load_state_dict(best_loss_state_dicts['feature_extractor'])
    classifier.load_state_dict(best_loss_state_dicts['classifier'])
    adversary_1.load_state_dict(best_loss_state_dicts['adversary_1'])
    adversary_2.load_state_dict(best_loss_state_dicts['adversary_2'])

    print("\n######## STARTING ADVERSARY FINETUNING LOOP #########")
    finetune_adversaries(args, train_loader, feature_extractor,
        adversary_1, adversary_2, adversary_optimizer)
    print("\n######## END ADVERSARY FINETUNING LOOP #########")
    
    print(f"\n######## START POST-FINETUNING VALIDATION #########")
    val_log, val_loss, val_accuracy = validate(
        args, val_loader, feature_extractor, classifier, adversary_1, adversary_2)
    print(f"Post-finetuning Validation Results")
    print_log(val_log)
    print(f"######## END POST-FINETUNING VALIDATION #########\n")
    
    # Save the models
    if args.save_dir is not None:
        print("Saving best-loss model...")
        state_dicts = {
            'feature_extractor': state_dict(feature_extractor),
            'classifier': state_dict(classifier),
            'adversary_1': state_dict(adversary_1),
            'adversary_2': state_dict(adversary_2)
        }
        save_dir = os.path.join(args.save_dir, save_subdir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(state_dicts, os.path.join(save_dir, 'best_accuracy.pth'))
    
    print("Loading best model based on validation accuracy...", end = "")
    feature_extractor.load_state_dict(best_accuracy_state_dicts['feature_extractor'])
    classifier.load_state_dict(best_accuracy_state_dicts['classifier'])
    adversary_1.load_state_dict(best_accuracy_state_dicts['adversary_1'])
    adversary_2.load_state_dict(best_accuracy_state_dicts['adversary_2'])

    print("\n######## STARTING ADVERSARY FINETUNING LOOP #########")
    finetune_adversaries(args, train_loader, feature_extractor,
        adversary_1, adversary_2, adversary_optimizer)
    print("\n######## END ADVERSARY FINETUNING LOOP #########")
    
    print(f"\n######## START POST-FINETUNING VALIDATION #########")
    val_log, val_loss, val_accuracy = validate(
        args, val_loader, feature_extractor, classifier, adversary_1, adversary_2)
    print(f"Post-finetuning Validation Results")
    print_log(val_log)
    print(f"######## END POST-FINETUNING VALIDATION #########\n")
    
    # Save the models
    if args.save_dir is not None:
        print("Saving best-accuracy model...")
        state_dicts = {
            'feature_extractor': state_dict(feature_extractor),
            'classifier': state_dict(classifier),
            'adversary_1': state_dict(adversary_1),
            'adversary_2': state_dict(adversary_2)
        }
        save_dir = os.path.join(args.save_dir, save_subdir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(state_dicts, os.path.join(save_dir, 'best_accuracy.pth'))

def main(args):
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = args.cudnn_benchmark

    transform = NullableWrapperTransform(torchvision.transforms.Compose([torchvision.transforms.Resize(223, max_size = 224), AdaptivePad(224), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    full_dataset = SVODataset(
        name='Custom',
        data_root='Custom',
        csv_path='Custom/annotations/dataset_v4_train.csv',
        transform = transform
    )

    subject_indices = []
    verb_indices = []
    object_indices = []
    for idx, (subject_image, verb_image, object_image, spatial_encodings, subject_label, verb_label, object_label) in enumerate(full_dataset):
        # Remove novel training examples
        if (subject_label is not None and subject_label.item() == 0) or (verb_label is not None and verb_label.item() == 0) or (object_label is not None and object_label.item() == 0):
            continue
        
        # Find non-null examples
        if subject_label is not None:
            subject_indices.append(idx)
        if verb_label is not None:
            verb_indices.append(idx)
        if object_label is not None:
            object_indices.append(idx)
    
    subject_dataset = SubjectImageSVODataset(torch.utils.data.Subset(full_dataset, subject_indices))
    verb_dataset = VerbImageSVODataset(torch.utils.data.Subset(full_dataset, verb_indices))
    object_dataset = ObjectImageSVODataset(torch.utils.data.Subset(full_dataset, object_indices))
    
    '''
    test_dataset = SVODataset(
        name='Custom',
        data_root='Custom',
        csv_path='Custom/annotations/dataset_v4_val.csv',
        transform = transform
    )

    # TODO filter the test set into subject, verb, and object classification
    # test sets, removing corresponding NoneType examples and removing all forms
    # of novelty.

    # TODO filter the test set into subject, verb, and object novelty detection
    # test sets, removing corresponding NoneType examples but keeping all forms
    # of novelty

    # TODO use the unfiltered test set as a novelty score correlation test set,
    # keeping all NoneType examples and all forms of novelty
    '''
    
    print('###############################')
    print('# Training subject classifier #')
    print('###############################')
    subject_classifier = torch.nn.Linear(args.bottleneck_dim, NUM_SUBJECTS).to(device)
    subject_verb_adversary = torch.nn.Linear(args.bottleneck_dim, NUM_VERBS).to(device)
    subject_object_adversary = torch.nn.Linear(args.bottleneck_dim, NUM_OBJECTS).to(device)
    run(args, subject_dataset, subject_classifier, subject_verb_adversary, subject_object_adversary, 'subject')
    
    print('################################')
    print('#   Training verb classifier   #')
    print('################################')
    verb_classifier = torch.nn.Linear(args.bottleneck_dim + args.spatial_encoding_dim, NUM_VERBS).to(device)
    verb_subject_adversary = torch.nn.Linear(args.bottleneck_dim, NUM_SUBJECTS).to(device)
    verb_object_adversary = torch.nn.Linear(args.bottleneck_dim, NUM_OBJECTS).to(device)
    run(args, verb_dataset, verb_classifier, verb_subject_adversary, verb_object_adversary, 'verb')

    print('################################')
    print('#  Training object classifier  #')
    print('################################')
    object_classifier = torch.nn.Linear(args.bottleneck_dim, NUM_OBJECTS).to(device)
    object_subject_adversary = torch.nn.Linear(args.bottleneck_dim, NUM_SUBJECTS).to(device)
    object_verb_adversary = torch.nn.Linear(args.bottleneck_dim, NUM_VERBS).to(device)
    run(args, object_dataset, object_classifier, object_subject_adversary, object_verb_adversary, 'object')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Application of DANN on the Sail-On dataset')
    parser.add_argument('--project-name',
                        default='DANN for Sail-On dataset',
                        type=str,
                        help='name of the project that appear on wandb.ai dashboard')
    parser.add_argument('--save-dir',
                        type=str,
                        required = True,
                        help='path to directory for model save files')
    parser.add_argument('--bottleneck-dim',
                        default=256,
                        type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--spatial-encoding-dim',
                        default=72,
                        type=int,
                        help='Dimension of flattened spatial encodings')
    parser.add_argument('--verb-trade-off',
                        default=0.5,
                        type=float,
                        help='the trade-off hyper-parameter for the verb classifying head')
    parser.add_argument('--object-trade-off',
                        default=0.5,
                        type=float,
                        help='the trade-off hyper-parameter for the object classifying head')
    # training hyperparameters
    parser.add_argument('-b',
                        '--train-batch-size',
                        default=32,
                        type=int,
                        metavar='N',
                        help='mini-batch size used for training (default: 32)')
    parser.add_argument('-be',
                        '--eval-batch-size',
                        default=64,
                        type=int,
                        metavar='N',
                        help='mini-batch size used for evaluation (validation and testing) (default: 64)')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--adversary-finetuning-epochs',
                        default=6000,
                        type=int,
                        help='number of total epochs to finetune adversaries after system training (default: 3000)')
    parser.add_argument('--lr',
                        default=0.01,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--adversary-lr',
                        default=0.01,
                        type=float,
                        metavar='ALR',
                        help='initial adversary learning rate')
    parser.add_argument('--adversary-step-ratio',
                        default=1.0,
                        type=float,
                        help='adversarial steps per non-adversarial step')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-3,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('--finetune', dest='finetune',
                        help='Use 0.1*lr on the backbone',
                        action='store_true')
    parser.add_argument('--no-finetune',
                        help='Use backbone learns as fast as the rest of the net',
                        dest='finetune', action='store_false')
    parser.set_defaults(finetune=True)
    # misc.
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
    parser.add_argument('--novel-category', dest='novel_category',
                        help='train with novel category',
                        action='store_true')
    parser.add_argument('--no-novel-category', dest='novel_category',
                        help='train with novel category',
                        action='store_false')
    parser.set_defaults(novel_category=False)
    parser.add_argument('--disable-adversaries',
                        dest='train_adversaries',
                        action='store_false')
    args = parser.parse_args()

    main(args)
