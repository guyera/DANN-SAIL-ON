from typing import List, Tuple

import torch
import torchvision.ops.boxes as box_ops
from torchvision.models.detection import transform

from data.data_factory import DataFactory

def compute_spatial_encodings(
        boxes_1: List[torch.Tensor], boxes_2: List[torch.Tensor],
        shapes: List[Tuple[int, int]], eps: float = 1e-10
) -> torch.Tensor:
    """
    Parameters:
    -----------
        boxes_1: List[torch.Tensor]
            First set of bounding boxes (M, 4)
        boxes_1: List[torch.Tensor]
            Second set of bounding boxes (M, 4)
        shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        eps: float
            A small constant used for numerical stability

    Returns:
    --------
        torch.Tensor
            Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape

        c1_x = (b1[:, 0] + b1[:, 2]) / 2
        c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2
        c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]
        b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]
        b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)

class HOINetworkTransform(transform.GeneralizedRCNNTransform):
    """
    Transformations for input image and target (box pairs)

    Arguments(Positional):
        min_size(int)
        max_size(int)
        image_mean(list[float] or tuple[float])
        image_std(list[float] or tuple[float])

    Refer to torchvision.models.detection for more details
    """

    def __init__(self, *args):
        super().__init__(*args)

    def resize(self, image, target):
        """
        Override method to resize box pairs
        """
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        scale_factor = min(
            self.min_size[0] / min_size,
            self.max_size / max_size
        )

        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor,
            mode='bilinear', align_corners=False,
            recompute_scale_factor=True
        )[0]
        if target is None:
            return image, target

        target['boxes_s'] = transform.resize_boxes(target['boxes_s'],
                                                   (h, w), image.shape[-2:])
        target['boxes_o'] = transform.resize_boxes(target['boxes_o'],
                                                   (h, w), image.shape[-2:])

        return image, target

    def postprocess(self, results, image_shapes, original_image_sizes):
        if self.training:
            loss = results.pop()

        for pred, im_s, o_im_s in zip(results, image_shapes, original_image_sizes):
            boxes_s, boxes_o = pred['boxes_s'], pred['boxes_o']
            boxes_s = transform.resize_boxes(boxes_s, im_s, o_im_s)
            boxes_o = transform.resize_boxes(boxes_o, im_s, o_im_s)
            pred['boxes_s'], pred['boxes_o'] = boxes_s, boxes_o

        if self.training:
            results.append(loss)

        return results

class SVODataset(torch.utils.data.Dataset):
    """
    Params:
        name: As in data.data_factory.DataFactory()
        data_root: As in data.data_factory.DataFactory()
        csv_path: As in data.data_factory.DataFactory()
        training: As in data.data_factory.DataFactory()
        min_size: As in HOINetworkTransform()
        max_size: As in HOINetworkTransform()
        image_mean: As in HOINetworkTransform()
        image_std: As in HOINetworkTransform()
    """
    def __init__(
            self,
            name,
            data_root,
            csv_path,
            training,
            min_size = 800,
            max_size = 1333,
            image_mean = None,
            image_std = None):
        super().__init__()
        self.dataset = DataFactory(
            name = name,
            data_root = data_root,
            csv_path = csv_path,
            training = training)
        
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        
        self.i_transform = HOINetworkTransform(
            min_size,
            max_size,
            image_mean,
            image_std
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        with torch.no_grad():
            image, detection, target = self.dataset[idx]
            original_image_size = image.shape[-2:]
            images, targets = self.i_transform([image], [target])
            image = images.tensors[0]
            target = targets[0]
            image_size = images.image_sizes[0]
            
            if detection['subject_boxes'][0][0].item() == -1:
                detection['subject_boxes'] = None
            else:
                detection['subject_boxes'] = transform.resize_boxes(detection['subject_boxes'], original_image_size, image_size)
            
            if detection['object_boxes'][0][0].item() == -1:
                detection['object_boxes'] = None
            else:
                detection['object_boxes'] = transform.resize_boxes(detection['object_boxes'], original_image_size, image_size)
            
            if target is None:
                subject_label = None
                object_label = None
                verb_label = None
            else:
                subject_label = target['subject'][0].detach()
                object_label = target['object'][0].detach()
                verb_label = target['verb'][0].detach()
                raw_subject_label = subject_label
                raw_object_label = object_label
                raw_verb_label = verb_label
                subject_label = None if raw_subject_label.item() == -1 else raw_subject_label
                object_label = None if raw_object_label.item() == -1 else raw_object_label
                verb_label = None if raw_subject_label.item() == -1 else raw_verb_label
            
            if detection['subject_boxes'] is not None and detection['object_boxes'] is not None:
                s_xmin, s_ymin, s_xmax, s_ymax = torch.round(detection['subject_boxes'][0]).to(torch.int)
                o_xmin, o_ymin, o_xmax, o_ymax = torch.round(detection['object_boxes'][0]).to(torch.int)
                v_xmin = min(s_xmin, o_xmin)
                v_ymin = min(s_ymin, o_ymin)
                v_xmax = max(s_xmax, o_xmax)
                v_ymax = max(s_ymax, o_ymax)
                
                x, y = torch.meshgrid(
                    torch.arange(1),
                    torch.arange(2)
                )
                x = x.flatten()
                y = y.flatten()
                coords = torch.cat([detection['subject_boxes'], detection['object_boxes']])
                
                spatial_encodings = compute_spatial_encodings(
                    [coords[x]], [coords[y]], [image_size]
                ).detach()
                subject_image = image[:, s_ymin : s_ymax, s_xmin : s_xmax]
                object_image = image[:, o_ymin : o_ymax, o_xmin : o_xmax]
                verb_image = image[:, v_ymin : v_ymax, v_xmin : v_xmax]
            elif detection['subject_boxes'] is not None:
                s_xmin, s_ymin, s_xmax, s_ymax = torch.round(detection['subject_boxes'][0]).to(torch.int)
                v_xmin = s_xmin
                v_ymin = s_ymin
                v_xmax = s_xmax
                v_ymax = s_ymax
                
                x, y = torch.meshgrid(
                    torch.arange(1),
                    torch.arange(2)
                )
                x = x.flatten()
                y = y.flatten()
                coords = torch.cat([detection['subject_boxes'], detection['subject_boxes']])
                
                spatial_encodings = compute_spatial_encodings(
                    [coords[x]], [coords[y]], [image_size]
                ).detach()
                subject_image = image[:, s_ymin : s_ymax, s_xmin : s_xmax]
                object_image = None
                verb_image = image[:, v_ymin : v_ymax, v_xmin : v_xmax]
            elif detection['object_boxes'] is not None:
                o_xmin, o_ymin, o_xmax, o_ymax = torch.round(detection['object_boxes'][0]).to(torch.int)
                
                spatial_encodings = None
                subject_image = None
                object_image = image[:, o_ymin : o_ymax, o_xmin : o_xmax]
                verb_image = None
            else:
                spatial_encodings = None
                subject_image = None
                object_image = None
                verb_image = None
            
            return subject_image, verb_image, object_image, spatial_encodings, subject_label, verb_label, object_label
