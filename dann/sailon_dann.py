import torch
import torchvision

from resnet import resnet18
from grl import GradientReverseLayer, WarmStartGradientReverseLayer
from data.svodataset import SVODataset
from featurizer import Featurizer
from classifier_head import ClassifierHead
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import List, Optional, Tuple
import os
import shutil
import argparse
import wandb
import numpy as np
from utils import accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def custom_collate(batch):
    subject_images = []
    verb_images = []
    object_images = []
    spatial_encodings = []
    subject_labels = []
    verb_labels = []
    object_labels = []
    for subject_image, verb_image, object_image, spatial_encoding, subject_label, verb_label, object_label in batch:
        subject_images.append(subject_image)
        verb_images.append(verb_image)
        object_images.append(object_image)
        spatial_encodings.append(spatial_encoding)
        subject_labels.append(subject_label)
        verb_labels.append(verb_label)
        object_labels.append(object_label)

    return subject_images, verb_images, object_images, spatial_encodings, subject_labels, verb_labels, object_labels


def train(train_loader: DataLoader, feature_extractor: Featurizer, classify_heads: List[Featurizer], optimizers: List[SGD]):
    subject_classifier, verb_discriminator, object_discriminator = classify_heads
    featurizer_opt, subject_opt, verb_opt, object_opt = optimizers
    
    feature_extractor.train()
    for head in classify_heads:
        head.train()

    subject_losses, subject_accuracies = [], []
    verb_losses, verb_accuracies = [], []
    object_losses, object_accuracies = [], []

    train_log = {}

    for subject_images, verb_images, object_images, spatial_encodings, subject_labels, verb_labels, object_labels in train_loader:
        subject_images = subject_images.to(device)
        verb_images = verb_images.to(device)
        object_images = object_images.to(device)
        subject_labels = subject_labels.to(device)
        verb_labels = verb_labels.to(device)
        object_labels = object_labels.to(device)

        '''Training with the subject images'''

        features = feature_extractor(subject_images)
        subject_preds = subject_classifier(features)
        subject_loss = F.cross_entropy(subject_preds, subject_labels)
        featurizer_opt.zero_grad()
        subject_opt.zero_grad()
        subject_loss.backward()
        featurizer_opt.step()
        subject_opt.step()

        '''Training with the verb images'''

        features = feature_extractor(verb_images)
        verb_preds = subject_classifier(features)
        verb_loss = F.cross_entropy(verb_preds, verb_labels)
        featurizer_opt.zero_grad()
        verb_opt.zero_grad()
        verb_loss.backward()
        featurizer_opt.step()
        verb_opt.step()

        '''Training with the object images'''

        features = feature_extractor(object_images)
        object_preds = subject_classifier(features)
        object_loss = F.cross_entropy(object_preds, object_labels)
        featurizer_opt.zero_grad()
        object_opt.zero_grad()
        object_loss.backward()
        featurizer_opt.step()
        object_opt.step()

        '''Logging losses and accuracies'''

        subject_losses.append(subject_loss)
        verb_losses.append(verb_losses)
        object_losses.append(object_losses)

        subject_accuracies.append(accuracy(subject_preds, subject_labels))
        verb_accuracies.append(accuracy(verb_preds, verb_labels))
        object_accuracies.append(accuracy(object_preds, object_labels))
    
    subject_losses = np.array(subject_losses)
    verb_losses = np.array(verb_losses)
    object_losses = np.array(object_losses)

    subject_accuracies = np.array(subject_accuracies)
    verb_accuracies = np.array(verb_accuracies)
    object_accuracies = np.array(object_accuracies)
    
    train_log.update({
        "Subject Classification Loss (Training Set)": subject_losses.mean(),
        "Verb Classification Loss (Training Set)": verb_losses.mean(),
        "Object Classification Loss (Training Set)": object_losses.mean(),

        "Subject Classification Accuracy (Training Set)": subject_accuracies.mean(),
        "Verb Classification Accuracy (Training Set)": verb_accuracies.mean(),
        "Object Classification Accuracy (Training Set)": object_accuracies.mean(),
    })

    return train_log


def validate():
    # TODO: validate the model on the validation dataset
    return 0

def evaluate():
    # TODO: post-training validation
    return 0    

def main(args):

    dataset = SVODataset(
        name = 'Custom',
        data_root = 'Custom',
        csv_path = 'Custom/annotations/dataset_v4_2_train.csv',
        training = True
    )

    # TODO: custom batch_size for dataloader
    # TODO: define a image transformation to reshape the image
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 16,
        shuffle = True,
        collate_fn = custom_collate
    )

    # TODO: define train_loader, val_loader, test_loader

    # init weights
    backbone = resnet18(pretrained=True)
    # TODO: define warm gradient layer to take up the entire training sequence
    # TODO: calculate the number of iterations
    max_iters = 1000
    warm_grl = WarmStartGradientReverseLayer(max_iters=max_iters)
    feature_extractor = Featurizer(backbone=backbone, bottleneck_dim=args.bottleneck_dim)
    subject_classifier = ClassifierHead(bottleneck_dim=args.bottleneck_dim)
    verb_discriminator = ClassifierHead(bottleneck_dim=args.bottleneck_dim, is_discriminator=True, grl=warm_grl)
    object_discriminator = ClassifierHead(bottleneck_dim=args.bottleneck_dim, is_discriminator=True, grl=warm_grl)

    # move all nets to device
    feature_extractor = feature_extractor.to(device)
    subject_classifier = subject_classifier.to(device)
    verb_discriminator = verb_discriminator.to(device)
    object_discriminator = object_discriminator.to(device)

    # optimizers
    featurizer_opt = SGD(feature_extractor.get_parameters(),
                        args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=True)

    subject_opt = SGD(subject_classifier.get_parameters(),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=True)

    verb_opt = SGD(verb_discriminator.get_parameters(),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=True)

    object_opt = SGD(object_discriminator.get_parameters(),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=True)
    
    
    classify_heads = [subject_classifier, verb_discriminator, object_discriminator]
    optimizers = [featurizer_opt, subject_opt, verb_opt, object_opt]
    best_acc = 0.0
    best_epoch = 0
    save_path = os.path.join('saved_models', args.project_name)    
    latest_path = os.path.join(save_path, 'latest')
    best_path = os.path.join(save_path, 'best')
    module_names = ['featurizer', 'subject', 'verb', 'object']
    

    for epoch in range(args.epochs):
        acc = train(train_loader, feature_extractor, classify_heads, optimizers)

        # save the latest module
        for module_name in module_names:
            torch.save(feature_extractor.state_dict(), os.path.join(latest_path, f'{module_name}.pth'))

        if acc > best_acc:
            best_epoch = epoch
            acc = best_acc
            # save the best model
            for module_name in module_names:
                shutil.copy(
                    os.path.join(latest_path, f'{module_name}.pth'),
                    os.path.join(best_path, f'{module_name}.pth'),
                )

    # TODO: post-training evaluation
            
        
        


        
        
        

        


    for subject_images, verb_images, object_images, spatial_encodings, subject_labels, verb_labels, object_labels in data_loader:
        for subject_image, verb_image, object_image, spatial_encoding, subject_label, verb_label, object_label in zip(subject_images, verb_images, object_images, spatial_encodings, subject_labels, verb_labels, object_labels):
            if subject_image is not None:
                print(f'Subject image shape: {subject_image.shape}')
            if verb_image is not None:
                print(f'Verb image shape: {verb_image.shape}')
            if object_image is not None:
                print(f'Object image shape: {object_image.shape}')
            if spatial_encoding is not None:
                print(f'Spatial encoding shape: {spatial_encoding.shape}')
            if subject_label is not None:
                print(f'Subject label shape: {subject_label.shape}')
            if verb_label is not None:
                print(f'Verb label shape: {verb_label.shape}')
            if object_label is not None:
                print(f'Object label shape: {object_label.shape}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Application of DANN on the Sail-On dataset')
    parser.add_argument('--bottleneck-dim',
                        default=256,
                        type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--subject-trade-off',
                        default=1.,
                        type=float,
                        help='the trade-off hyper-parameter for transfer loss with the subject head')
    parser.add_argument('--subject-trade-off',
                        default=1.,
                        type=float,
                        help='the trade-off hyper-parameter for transfer loss with the subject head')