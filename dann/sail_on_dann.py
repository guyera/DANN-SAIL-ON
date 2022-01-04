import torch
import torchvision

from resnet import resnet18
from grl import GradientReverseLayer, WarmStartGradientReverseLayer
from data.svodataset import SVODataset
from featurizer import Featurizer
from classifier_head import ClassifierHead
from torch.optim import SGD


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


def train():
    pass


def main(args):

    dataset = SVODataset(
        name = 'Custom',
        data_root = 'Custom',
        csv_path = 'Custom/annotations/dataset_v4_2_train.csv',
        training = True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 16,
        shuffle = True,
        collate_fn = custom_collate
    )

    # init weights
    backbone = resnet18(pretrained=True)
    warm_grl = WarmStartGradientReverseLayer()
    feature_extractor = Featurizer(backbone=backbone, bottleneck_dim=args.bottleneck_dim)
    subject_classifier = ClassifierHead(bottleneck_dim=args.bottleneck_dim)
    verb_discriminator = ClassifierHead(bottleneck_dim=args.bottleneck_dim, is_discriminator=True, grl=warm_grl)
    object_discriminator = ClassifierHead(bottleneck_dim=args.bottleneck_dim, is_discriminator=True, grl=warm_grl)

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
