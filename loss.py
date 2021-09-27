import torch
from torch import nn, Tensor


class IoULoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_box: Tensor, gt_box: Tensor):
        xmin_inter = torch.maximum(gt_box[:, 0], pred_box[:, 0])
        ymin_inter = torch.maximum(gt_box[:, 1], pred_box[:, 1])
        xmax_inter = torch.minimum(gt_box[:, 2], pred_box[:, 2])
        ymax_inter = torch.minimum(gt_box[:, 3], pred_box[:, 3])
        w_inter = torch.clamp(xmax_inter - xmin_inter, min=0)
        h_inter = torch.clamp(ymax_inter - ymin_inter, min=0)
        inter_area = w_inter * h_inter
        gt_box_area = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])
        pred_box_area = (pred_box[:, 2] - pred_box[:, 0]) * (pred_box[:, 3] - pred_box[:, 1])
        union_area = gt_box_area + pred_box_area - inter_area
        iou = inter_area / union_area
        # iou = torch.where(iou != 0, -torch.log(iou), iou)
        loss = 1 - iou
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return torch.mean(loss)


class GIoULoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_box: Tensor, gt_box: Tensor):
        xmin_inter = torch.maximum(gt_box[:, 0], pred_box[:, 0])
        ymin_inter = torch.maximum(gt_box[:, 1], pred_box[:, 1])
        xmax_inter = torch.minimum(gt_box[:, 2], pred_box[:, 2])
        ymax_inter = torch.minimum(gt_box[:, 3], pred_box[:, 3])
        w_inter = torch.clamp(xmax_inter - xmin_inter, min=0)
        h_inter = torch.clamp(ymax_inter - ymin_inter, min=0)
        inter_area = w_inter * h_inter
        gt_box_area = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])
        pred_box_area = (pred_box[:, 2] - pred_box[:, 0]) * (pred_box[:, 3] - pred_box[:, 1])
        union_area = gt_box_area + pred_box_area - inter_area
        iou = inter_area / union_area
        # iou = torch.where(iou != 0, -torch.log(iou), iou)
        # c_area is the smallest box enclosing the two bounding boxes
        xmin_c = torch.minimum(gt_box[:, 0], pred_box[:, 0])
        ymin_c = torch.minimum(gt_box[:, 1], pred_box[:, 1])
        xmax_c = torch.maximum(gt_box[:, 2], pred_box[:, 2])
        ymax_c = torch.maximum(gt_box[:, 3], pred_box[:, 3])
        c_area = (xmax_c - xmin_c) * (ymax_c - ymin_c)
        giou = iou - ((c_area - union_area) / c_area)
        loss = 1 - giou
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return torch.mean(loss)
