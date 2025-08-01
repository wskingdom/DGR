#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from DGR import DGR
from params import ModelParams


class CTGeometry:

    gaussians : DGR

    def __init__(self, args : ModelParams, gaussians : DGR, load_iteration=None, shuffle=True, resolution_scales=[1.0], mode='svct'):
        """b
        :param path: Path to colmap scene main folder.
        """
        assert mode in ['svct', 'lact', 'cbct']
        self.model_path = args.model_path
        self.source_path = args.source_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            if mode == 'svct':
                self.gaussians.create_from_svct(self.source_path, None)
            elif mode == 'lact':
                self.gaussians.create_from_lact(self.source_path, None)
            elif mode == 'cbct':
                self.gaussians.create_from_svct(self.source_path, process='tigre')

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]