# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import namedtuple

import torch
import torch.nn as nn

from smplx.lbs import transform_mat


PerspParams = namedtuple('ModelOutput',
                         ['rotation', 'translation', 'center',
                          'focal_length'])


def create_camera(camera_type='persp', **kwargs):
    if camera_type.lower() == 'persp':
        return PerspectiveCamera(**kwargs)
    elif camera_type.lower() == 'calib':
        return CalibratedCamera(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(camera_type))


class CalibratedCamera(nn.Module):

    def __init__(self,
                 calibs=None,
                 scale=None,
                 translation=None,
                 perspective=False,
                 batch_size=1,
                 dtype=torch.float32,
                 **kwargs) -> None:
        """Camera with calibrated matrix.

        Args:
            calibs (tensor): BxVx4x4, calibrated matrix of views.
            scale (_type_, optional): _description_. Defaults to None.
            translate (_type_, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 1.
            dtype (_type_, optional): _description_. Defaults to torch.float32.
        """
        super().__init__()
        if scale is None:
            scale = torch.ones([batch_size, 1], dtype=dtype)
        scale = nn.Parameter(scale, requires_grad=True)
        self.register_parameter('scale', scale)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)
        translation = nn.Parameter(translation,
                                   requires_grad=True)
        self.register_parameter('translation', translation)

        if calibs is not None:
            self.set_calibration_matrix(calibs)

        self.perspective = perspective
        self.dtype = dtype
        self.batch_size = batch_size
    
    def set_calibration_matrix(self, calibs):
        assert calibs.shape[0] == self.batch_size
        calibs = torch.tensor(calibs, dtype=self.dtype)
        calibs = calibs.to(self.scale.device)
        self.register_buffer('calibs', calibs)
        self.num_views = calibs.shape[1]
    
    def forward(self, points):
        """Forward pass.

        Args:
            points (tensor): BxNx3.
        """
        points = points * self.scale[:, None, :] + self.translation[:, None, :]
        points = points[:, None, ...].expand(-1, self.num_views, -1, -1)
        points = torch.cat(
            [points, torch.ones_like(points[..., :1])], dim=-1)
        points = points @ self.calibs.transpose(-1, -2) # (B, V, N, 4) @ (B, V, 4, 4)

        if self.perspective:
            points = points[..., :3]
            # Here, we assume the calibration matrix do not contain inverse z.
            # If not, just comment the following line.
            points[..., 2] = -points[..., 2]
            points[..., 0] /= points[..., 2]
            points[..., 1] /= points[..., 2]
        else:
            points = points[..., :2]
        return points


class PerspectiveCamera(nn.Module):

    FOCAL_LENGTH = 5000

    def __init__(self, rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_x is None else
                focal_length_x,
                dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_y is None else
                focal_length_y,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        if rotation is None:
            rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)

        rotation = nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)

        translation = nn.Parameter(translation,
                                   requires_grad=True)
        self.register_parameter('translation', translation)

    def forward(self, points):
        device = points.device

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points
