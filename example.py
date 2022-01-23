import torch
import torchvision

from data.svodataset import SVODataset


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


dataset = SVODataset(
    name='Custom',
    data_root='Custom',
    csv_path='Custom/annotations/dataset_v4_2_train.csv',
    training=True
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=custom_collate
)

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

    for subject_image, verb_image, object_image, spatial_encoding, subject_label, verb_label, object_label in zip(subject_images, verb_images, object_images, spatial_encodings, subject_labels, verb_labels, object_labels):
        if subject_image is not None and verb_image is not None and object_image is not None:
            to_pil_image = torchvision.transforms.ToPILImage()
            subject_image_pil = to_pil_image(subject_image)
            verb_image_pil = to_pil_image(verb_image)
            object_image_pil = to_pil_image(object_image)
            subject_image_pil.save('some_images/subject_example_image.jpg')
            verb_image_pil.save('some_images/verb_example_image.jpg')
            object_image_pil.save('some_images/object_example_image.jpg')
            # NOTE: These example images will look corrupted. That's because
            # they're standardized to have mean 0 std 1 for each color channel,
            # and PIL isn't aware of this, so it has no way of "unstandardizing"
            # them to get the raw pixel data. But you can generally still make
            # out the content of the images to some extent
            break

    break
