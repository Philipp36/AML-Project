import torch
import math
import random
from collections.abc import Sequence
from torchvision.transforms import functional as F, CenterCrop, RandomRotation, InterpolationMode
from torch import nn


class VerticalFlip(nn.Module):
    """
       Mirror the image and the corresponding heatmap along the y-Axis.
    """
    @torch.no_grad()
    def forward(self, img, heat):
        """
         Args:
             img (PIL Image or Tensor): Image to be flipped.
             heat Tensor: Heatmap to be flipped.

         Returns:
             PIL Image or Tensor: Flipped image.
             heat Tensor: Flipped heatmap
         """
        # img = F.vflip(img)
        # xmin, ymin, xmax, ymax = box
        # _, img_h = F._get_image_size(img)
        # tmp = img_h - ymin
        # ymin = img_h - ymax
        # ymax = tmp
        # return img, (xmin, ymin, xmax, ymax)
        return F.vflip(img), F.vflip(heat)


class HorizontalFlip(nn.Module):
    """
          Mirror the image and the corresponding heatmap along the x-Axis.
    """
    @torch.no_grad()
    def forward(self, img, heat):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
            heat Tensor: Heatmap to be flipped.

        Returns:
            PIL Image or Tensor: Flipped image.
            heat Tensor: Flipped heatmap
        """
        #img = F.hflip(img)
        #xmin, ymin, xmax, ymax = box
        #img_w, _ = F._get_image_size(img)
        #tmp = img_w - xmin
        #xmin = img_w - xmax
        #xmax = tmp
        #return img, (xmin, ymin, xmax, ymax)
        return F.hflip(img), F.hflip(heat)


class CenterCrop(CenterCrop):
    """
    Crop Image and Heatmap to (size_w, size_h).
    """
    def __init__(self, size):
        assert size[0] == size[1]
        super().__init__(size)

    @torch.no_grad()
    def forward(self, img, heat):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            heat: Box to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
            heat Tensor: Cropped heatmap
        """
        img_w, img_h = F._get_image_size(img)
        # resize_factor_x = self.size[0] / img_w
        # resize_factor_y = self.size[1] / img_h
        # [sx, sy, ex, ey] = box
        # sx -= ((1 - resize_factor_x) * img_w) // 2
        # ex -= ((1 - resize_factor_x) * img_w) // 2
        # sy -= ((1 - resize_factor_y) * img_h) // 2
        # ey -= ((1 - resize_factor_y) * img_h) // 2
        # box = rescale_box([sx, sy, ex, ey], self.size, F._get_image_size(img))
        return F.resize(super().forward(img), (img_h, img_w)),\
            F.resize(super().forward(heat), (img_h, img_w)),


class RandomRotation(RandomRotation):
    def __init__(self, degrees, interpolation=InterpolationMode.NEAREST, expand=False,
                 center=None, fill=0, resample=None):
        super().__init__(degrees, interpolation, expand, center, resample)

    @torch.no_grad()
    def forward(self, img, heat):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.
            heat: Box to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
            heat Tensor: Rotated heatmap
        """

        fill_img = fill_heat = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill_img, (int, float)):
                fill = [float(fill_img)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill_img]
        if isinstance(heat, torch.Tensor):
            if isinstance(fill_heat, (int, float)):
                fill = [float(fill_heat)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill_heat]
        angle = self.get_params(self.degrees)
        # img_w, img_h = F._get_image_size(img)
        return F.rotate(img, angle, self.resample, self.expand, self.center, fill_img), \
            F.rotate(heat, angle, self.resample, self.expand, self.center, fill_heat)
            #rotate_bb(box, angle, img_w // 2, img_h // 2)


class RandomChoice(nn.Module):
    """Apply single transformation with a probability of p randomly picked from a list.
        Optionally with custom probability distribution.
       """

    def __init__(self, transforms, p_transforms=None, p=1):
        if p_transforms is not None and not isinstance(p_transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        assert 0 <= p <= 1
        self.p_transforms = p_transforms
        self.transforms = transforms
        self.p = p

    def __call__(self, *args):
        t = random.choices(self.transforms, weights=self.p_transforms)[0]
        return t(*args) if torch.rand(1) < self.p else args

    def __repr__(self):
        format_string = super().__repr__()
        format_string += '(p_transforms={0}, p={1})'.format(self.p_transforms, self.p)
        return format_string
