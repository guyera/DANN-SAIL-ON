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
import torch.backends.cudnn as cudnn
from typing import List, Optional, Tuple
import os
import shutil
import argparse
import wandb
import numpy as np
from utils import accuracy, print_log, to_torch_batch, clean_batch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("GPU detected")
else:
    print("No GPU dectected, defaulting to CPU")


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


def train(args: argparse.Namespace, train_loader: DataLoader, feature_extractor: Featurizer, classify_heads: List[Featurizer], optimizers: List[SGD]):
    subject_classifier, verb_discriminator, object_discriminator = classify_heads
    featurizer_opt, subject_opt, verb_opt, object_opt = optimizers

    feature_extractor.train()
    for head in classify_heads:
        head.train()

    subject_losses, subject_accuracies = [], []
    verb_losses, verb_accuracies = [], []
    object_losses, object_accuracies = [], []

    train_log = {}
    count = 0

    for subject_images, verb_images, object_images, spatial_encodings, subject_labels, verb_labels, object_labels in train_loader:
        clean_batch(subject_images, verb_images, object_images,
                    subject_labels, verb_labels, object_labels)
        subject_images = to_torch_batch(subject_images, device)
        verb_images = to_torch_batch(verb_images, device)
        object_images = to_torch_batch(object_images, device)
        subject_labels = to_torch_batch(subject_labels, device)
        # subject_labels = subject_labels.to(device)
        verb_labels = to_torch_batch(verb_labels, device)
        object_labels = to_torch_batch(object_labels, device)

        # Training with the subject images

        features = feature_extractor(subject_images)
        subject_preds = subject_classifier(features)
        subject_loss = F.cross_entropy(subject_preds, subject_labels)
        featurizer_opt.zero_grad()
        subject_opt.zero_grad()
        subject_loss.backward()
        featurizer_opt.step()
        subject_opt.step()

        # Training with the verb images

        features = feature_extractor(verb_images)
        verb_preds = verb_discriminator(features)
        verb_loss = F.cross_entropy(verb_preds, verb_labels)
        featurizer_opt.zero_grad()
        verb_opt.zero_grad()
        verb_loss.backward()
        featurizer_opt.step()
        verb_opt.step()

        # Training with the object images

        features = feature_extractor(object_images)
        object_preds = object_discriminator(features)
        object_loss = F.cross_entropy(object_preds, object_labels)
        featurizer_opt.zero_grad()
        object_opt.zero_grad()
        object_loss.backward()
        featurizer_opt.step()
        object_opt.step()

        # Logging losses and accuracies

        subject_losses.append(subject_loss.item())
        verb_losses.append(verb_loss.item())
        object_losses.append(object_loss.item())

        subject_accuracy = accuracy(subject_preds, subject_labels)
        verb_accuracy = accuracy(verb_preds, verb_labels)
        object_accuracy = accuracy(object_preds, object_labels)

        subject_accuracies.append(subject_accuracy)
        verb_accuracies.append(verb_accuracy)
        object_accuracies.append(object_accuracy)

        if count % args.print_freq == 0:
            print(f"Results from training batch {count}: ")
            print(
                f"Subject: Loss = {subject_loss}, Accuracy = {subject_accuracy}%")
            print(f"Verb: Loss = {verb_loss}, Accuracy = {verb_accuracy}%")
            print(
                f"Object: Loss = {object_loss}, Accuracy = {object_accuracy}%")
            print('\n')

        count += 1

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


def validate(args: argparse.Namespace, val_loader: DataLoader, feature_extractor: Featurizer, classify_heads: List[Featurizer], dataset_type: Optional[str] = 'Validaton'):
    subject_classifier, verb_discriminator, object_discriminator = classify_heads

    feature_extractor.eval()
    for head in classify_heads:
        head.eval()

    subject_losses, subject_accuracies = [], []
    verb_losses, verb_accuracies = [], []
    object_losses, object_accuracies = [], []

    val_log = {}
    count = 0

    with torch.no_grad():
        for subject_images, verb_images, object_images, spatial_encodings, subject_labels, verb_labels, object_labels in val_loader:
            subject_images = subject_images.to(device)
            verb_images = verb_images.to(device)
            object_images = object_images.to(device)
            subject_labels = subject_labels.to(device)
            verb_labels = verb_labels.to(device)
            object_labels = object_labels.to(device)

            subject_features = feature_extractor(subject_images)
            verb_features = feature_extractor(verb_images)
            object_features = feature_extractor(object_images)

            subject_preds = subject_classifier(subject_features)
            verb_preds = verb_discriminator(verb_features)
            object_preds = object_discriminator(object_features)

            subject_loss = F.cross_entropy(subject_preds, subject_labels)
            verb_loss = F.cross_entropy(verb_preds, verb_labels)
            object_loss = F.cross_entropy(object_preds, object_labels)

            subject_accuracy = accuracy(subject_preds, subject_labels)
            verb_accuracy = accuracy(verb_preds, verb_labels)
            object_accuracy = accuracy(object_preds, object_labels)

            subject_losses.append(subject_loss.item())
            verb_losses.append(verb_loss.item())
            object_losses.append(object_loss.item())

            subject_accuracies.append(subject_accuracy)
            verb_accuracies.append(verb_accuracy)
            object_accuracies.append(object_accuracy)

            if count % args.print_freq == 0:
                print(f"Results from {dataset_type} batch {count}: ")
                print(
                    f"Subject: Loss = {subject_loss}, Accuracy = {subject_accuracy}%")
                print(f"Verb: Loss = {verb_loss}, Accuracy = {verb_accuracy}%")
                print(
                    f"Object: Loss = {object_loss}, Accuracy = {object_accuracy}%")
                print('\n')

            count += 1

    subject_losses = np.array(subject_losses)
    verb_losses = np.array(verb_losses)
    object_losses = np.array(object_losses)

    subject_accuracies = np.array(subject_accuracies)
    verb_accuracies = np.array(verb_accuracies)
    object_accuracies = np.array(object_accuracies)

    val_log.update({
        f"Subject Classification Loss ({dataset_type} Set)": subject_losses.mean(),
        f"Verb Classification Loss ({dataset_type} Set)": verb_losses.mean(),
        f"Object Classification Loss ({dataset_type} Set)": object_losses.mean(),

        f"Subject Classification Accuracy ({dataset_type} Set)": subject_accuracies.mean(),
        f"Verb Classification Accuracy ({dataset_type} Set)": verb_accuracies.mean(),
        f"Object Classification Accuracy ({dataset_type} Set)": object_accuracies.mean(),
    })

    return val_log, subject_accuracies.mean()


def test(args: argparse.Namespace, test_loader: DataLoader, feature_extractor: Featurizer, classify_heads: List[Featurizer], dataset_type: Optional[str] = 'Test'):
    # TODO: post-training validation
    # TODO: same evaluation as OfficeHome dataset (calibration, accuracy)
    test_log = validate(args, test_loader, feature_extractor,
                        classify_heads, dataset_type)
    return test_log


def main(args):

    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = args.cudnn_benchmark

    full_dataset = SVODataset(
        name='Custom',
        data_root='Custom',
        csv_path='Custom/annotations/dataset_v4_2_train.csv',
        training=True,
        max_size=224
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    test_dataset = SVODataset(
        name='Custom',
        data_root='Custom',
        csv_path='Custom/annotations/dataset_v4_2_val.csv',
        training=True,  # TODO: figure out what this is
        max_size=224
    )

    # TODO: define a image transformation to reshape the image
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

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )

    # init weights
    backbone = resnet18(pretrained=True)
    # define gradient layer with scheduled trade-off parameter for the verb and object discriminator
    max_iters = len(train_loader) * args.epochs
    verb_grl = WarmStartGradientReverseLayer(
        hi=args.verb_trade_off, max_iters=max_iters)
    object_grl = WarmStartGradientReverseLayer(
        hi=args.object_trade_off, max_iters=max_iters)
    feature_extractor = Featurizer(
        backbone=backbone, bottleneck_dim=args.bottleneck_dim)
    subject_classifier = ClassifierHead(
        num_classes=5, bottleneck_dim=args.bottleneck_dim)
    verb_discriminator = ClassifierHead(
        num_classes=8, bottleneck_dim=args.bottleneck_dim, is_discriminator=True, grl=verb_grl)
    object_discriminator = ClassifierHead(
        num_classes=12, bottleneck_dim=args.bottleneck_dim, is_discriminator=True, grl=object_grl)

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

    classify_heads = [subject_classifier,
                      verb_discriminator, object_discriminator]
    optimizers = [featurizer_opt, subject_opt, verb_opt, object_opt]
    best_acc = 0.0
    best_epoch = 0
    latest_path = os.path.join(args.project_folder, 'latest')
    best_path = os.path.join(args.project_folder, 'best')
    module_names = ['featurizer', 'subject', 'verb', 'object']

    print("######## STARTING TRAINING LOOP #########")
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")
        train_log = train(args, train_loader, feature_extractor,
                          classify_heads, optimizers)
        print(f"Epoch {epoch} Training Results")
        wandb.log(train_log)
        print_log(train_log)

        val_log, acc = validate(
            args, val_loader, feature_extractor, classify_heads)
        print(f"Epoch {epoch} Validation Results")
        wandb.log(val_log)
        print_log(val_log)

        # save the latest module
        for module_name in module_names:
            torch.save(feature_extractor.state_dict(),
                       os.path.join(latest_path, f'{module_name}.pth'))

        if acc > best_acc:
            best_epoch = epoch
            best_acc = acc
            # save the best model
            for module_name in module_names:
                shutil.copy(
                    os.path.join(latest_path, f'{module_name}.pth'),
                    os.path.join(best_path, f'{module_name}.pth'),
                )
    wandb.log({'best training epoch': best_epoch})
    print("######## ENDING TRAINING LOOP #########\n")

    print("######## STARTING EVALUATION #########")
    # Load the best model for evaluation
    for module, module_name in ([feature_extractor] + classify_heads), module_names:
        module.load_state_dict(
            torch.load(os.path.join(best_path, f'{module_name}.pth')))
    test_log, test_acc = test(
        args, test_loader, feature_extractor, classify_heads)
    wandb.log(test_log)
    print_log(test_log)
    print("######## ENDING EVALUATION #########\n")
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Application of DANN on the Sail-On dataset')
    parser.add_argument('--project-name',
                        default='DANN for Sail-On dataset',
                        type=str,
                        help='name of the project that appear on wandb.ai dashboard')
    parser.add_argument('--bottleneck-dim',
                        default=256,
                        type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--verb-trade-off',
                        default=1.,
                        type=float,
                        help='the trade-off hyper-parameter for the verb classifying head')
    parser.add_argument('--object-trade-off',
                        default=1.,
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
                        default=30,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run (default: 30)')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.01,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
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
    # misc.
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-p',
                        '--print-freq',
                        default=100,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cudnn-benchmark',
                        default=True,
                        type=bool,
                        help='flag for torch.cudnn_benchmark')
    args = parser.parse_args()
    wandb.login()
    wandb.init(wandb.init(project=args.project_name))
    args.project_folder = os.path.join(
        'saved_models', wandb.run.project, wandb.run.name)
    if not os.path.isdir(args.project_folder):
        os.makedirs(args.project_folder)
    # update the args with the sweep configurations
    if wandb.run:
        wandb.config.update({k: v for k, v in vars(
            args).items() if k not in wandb.config.as_dict()})
        args = argparse.Namespace(**wandb.config.as_dict())
    main(args)
