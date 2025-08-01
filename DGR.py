import torch
import torch.nn.functional as F
from odl.contrib import torch as odl_torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import odl
import cv2
from tqdm import tqdm
from utils.previous_utils import read_image
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.utils import load_reconstruction, quick_evaluation, count_neighbors_gpu, resize_and_crop_sr, resize_volume, resize_volume2, compute_mean_distance
from utils.odl_utils import ConeBeam3DProjector
from utils.cbct_utils import init_geometry
import tigre
import tigre.algorithms as algs

def build_isotropic_covariance(sigma):
    """
    Creates an isotropic covariance matrix for 3D Gaussian splatting.

    Args:
        sigma: The desired standard deviation (same for all axes).

    Returns:
        A 3x3 isotropic covariance matrix.
    """
    isotropic_variance = sigma ** 2  # Square sigma to get variance
    covariance = torch.eye(3).cuda() * isotropic_variance.unsqueeze(1)
    return covariance

class DGR:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            std = scaling * scaling_modifier
            covariance = build_isotropic_covariance(std)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm, actual_covariance, covariance, std
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.image_size = 256
        self.n_view = 50
        self.img_path = '/sharedata/datasets/real/real_dataset/cone_ntrain_50_angle_360/pine/vol_gt.npy'
        # self.img_path = 'L067.npy'

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def count_neighbors(points, distance_threshold):
        """
        Counts the number of neighbors for each point in a 3D space.

        Args:
            points: A NumPy array of shape [N, 3], where N is the number of points
                    and each row represents a point's coordinates (x, y, z).
            distance_threshold: The maximum distance considered as a neighbor.

        Returns:
            A NumPy array of shape [N] containing the number of neighbors for each point.
        """

        # Calculate pairwise squared distances for efficiency
        squared_distances = np.linalg.norm(points[:, None] - points[None], axis=2) ** 2

        # Create a mask to exclude points from their own neighborhood and diagonal elements
        mask = np.ones_like(squared_distances, dtype=bool)
        np.fill_diagonal(mask, False)

        # Count neighbors based on distance threshold
        neighbors = np.count_nonzero(squared_distances[mask] <= distance_threshold**2, axis=1)

        return neighbors
    
    
    def create_from_svct(self, test_id='', spatial_lr_scale=0, process='odl'):
        self.vis_dir = 'vis_svct'
        if process == 'odl':
            view_number = self.n_view
            self.n_view = int(view_number)
            Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, self.n_view)
            Fan_detector_partition = odl.uniform_partition(-360, 360, 720)
            Fan_geometry = odl.tomo.FanBeamGeometry(Fan_angle_partition, Fan_detector_partition,
                                        src_radius=500, det_radius=500)
            Fan_reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[self.image_size, self.image_size], dtype='float32')
            Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry, impl='astra_cuda')
            Fan_FBP = odl.tomo.fbp_op(Fan_ray_trafo)
            Fan_filter = odl.tomo.fbp_filter_op(Fan_ray_trafo)
            self.Fan_ray_trafo = odl_torch.OperatorModule(Fan_ray_trafo) 
            self.Fan_FBP = odl_torch.OperatorModule(Fan_FBP)      
            self.vis_dir = f'vis_{os.path.basename(test_id)}_{self.n_view}'
            img_path = self.img_path
            volume = np.load(img_path, allow_pickle=True)
            if volume.max() <= 1:
                volume = volume * 255
            self.image_size = (volume.shape[0], self.image_size, self.image_size)
            self.scene_extent = np.linalg.norm(self.image_size) / self.image_size[1]
            volume = resize_volume2(volume, self.image_size[1:], save_root='input') / 255
            low_sinogram = self.Fan_ray_trafo(volume)
            self.gt_projs = low_sinogram
            self.gt_image = volume
            # low_sinogram = F.interpolate(low_sinogram[None,...], size=(720, 720), mode='bilinear', align_corners=False)[0]
            I_FBP = self.Fan_FBP(low_sinogram)  # [bs, n, h, w] -> [bs, x, y, z]
            sample_img = I_FBP[8].cpu().numpy()*255
            cv2.imwrite(f'{self.vis_dir}/FBP.png', sample_img)
            print(low_sinogram.shape)
            print(I_FBP.shape)
            quick_evaluation(I_FBP, volume)
        elif process == 'tigre':
            # cone beam ct
            img_path = self.img_path
            # img_path = 'gt_256.npy'
            volume = np.load(img_path, allow_pickle=True)
            if volume.max() > 1:
                volume = volume / 255
            geo, angles = init_geometry(n_sample=self.n_view)
            self.geo = geo
            self.angles = angles
            self.image_size = (volume.shape[0], self.image_size, self.image_size)
            self.scene_extent = np.linalg.norm(self.image_size) / self.image_size[1]
            sparse_projection = tigre.Ax(volume, geo, angles)
            I_FBP = algs.fdk(sparse_projection, geo, angles)
            I_FBP = torch.from_numpy(I_FBP)
            volume = torch.from_numpy(volume)
            quick_evaluation(I_FBP, volume)
            print(sparse_projection.shape)
            self.gt_projs = torch.from_numpy(sparse_projection).cuda()
            self.gt_image = volume.cuda()
        else:
            raise NotImplementedError
        grad_maximum = 1e-5 # the upper bound for the gradient
        intensity_threshold = 0.001 # the lower bound for intensity
        fbp_threshold = 0.05 # threshold for the FBP image
        n_gaussian = 200000 # number of initail gaussians 150K-400K. 
        max_gaussian = 250000
        distance_threshold = 0.1 # threshold for neighbor distance
        radii_coefficient = 0.12 # coefficient for the radii of the gaussians
        intensity_coefficient = 0.3 # coefficient for the intensity of the gaussians
        
        I_FBP = I_FBP.cpu()
        self.volume_shape = I_FBP.shape
        self.max_gaussian = max_gaussian
        # filter I_FBP lower than the threshold
        I_FBP[I_FBP < fbp_threshold] = 0
        # compute the gradient of I_FBP
        grad_I_FBP = np.gradient(I_FBP)
        # compute the norm of the gradient
        # grad_norm = np.sqrt(grad_I_FBP[0]**2 + grad_I_FBP[1]**2 + grad_I_FBP[2]**2)
        grad_norm = np.linalg.norm(grad_I_FBP, axis=0, keepdims=True)[0]
        # rank the coordinates based on the grad_norm
        median = np.median(grad_norm[grad_norm > 0])
        distance = np.abs(grad_norm - median)
        median_indices = np.argsort(distance.reshape(-1))[:n_gaussian]
        median_indices = np.unravel_index(median_indices, grad_norm.shape)
        selected_norm_value = grad_norm[median_indices] # just used for debug. They should be near the median value
        points = np.stack(median_indices, axis=-1)
        # normalize the points into [0, 1]
        points = points / np.array(I_FBP.shape)
        colors = I_FBP[median_indices].unsqueeze(dim=1)
        
        # n_neighbours = count_neighbors(points,distance_threshold)
        n_neighbours = count_neighbors_gpu(points, distance_threshold)
        radii = radii_coefficient / n_neighbours
        colors = intensity_coefficient * colors
        
        colors[colors < intensity_threshold] = intensity_threshold

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        # fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
        fused_color = torch.tensor(np.asarray(colors)).float().cuda()
        features = torch.zeros((fused_color.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :1, 0 ] = fused_color
        features[:, 1:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist = torch.clamp_min(compute_mean_distance(torch.from_numpy(np.asarray(points)).float().cuda()), 1e-7)
        scales = torch.log(torch.sqrt(dist))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_radii2D = torch.from_numpy(radii).cuda()




    def create_from_lact(self, source_path : str, spatial_lr_scale : float, process='odl'):
        self.vis_dir = 'vis_lact'
        if process == 'odl':
            self.n_view = 720 # full view, and then need to limit the angle
            Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, self.n_view)
            Fan_detector_partition = odl.uniform_partition(-360, 360, 720)
            Fan_geometry = odl.tomo.FanBeamGeometry(Fan_angle_partition, Fan_detector_partition,
                                        src_radius=500, det_radius=500)
            Fan_reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[self.image_size, self.image_size], dtype='float32')
            Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry, impl='astra_cuda')
            Fan_FBP = odl.tomo.fbp_op(Fan_ray_trafo)
            Fan_filter = odl.tomo.fbp_filter_op(Fan_ray_trafo)
            self.Fan_ray_trafo = odl_torch.OperatorModule(Fan_ray_trafo) 
            self.Fan_FBP = odl_torch.OperatorModule(Fan_FBP)      
            img_path = self.img_path
            volume = np.load(img_path, allow_pickle=True)
            if volume.max() <= 1:
                volume = volume * 255
            self.image_size = (volume.shape[0], self.image_size, self.image_size)
            self.scene_extent = np.linalg.norm(self.image_size) / self.image_size[1]
            volume = resize_volume2(volume, self.image_size[1:], save_root='input') / 255
            low_sinogram = self.Fan_ray_trafo(volume)
            low_sinogram[:, low_sinogram.shape[1]//2:] = 0 # limit the angle of view
            self.gt_projs = low_sinogram
            self.gt_image = volume
            # low_sinogram = F.interpolate(low_sinogram[None,...], size=(720, 720), mode='bilinear', align_corners=False)[0]
            I_FBP = self.Fan_FBP(low_sinogram)  # [bs, n, h, w] -> [bs, x, y, z]
            sample_img = I_FBP[8].cpu().numpy()*255
            resp = cv2.imwrite(f'{self.vis_dir}/FBP.png', sample_img)
            print(resp)
            print(low_sinogram.shape)
            print(I_FBP.shape)
            quick_evaluation(I_FBP, volume)
        else:
            raise NotImplementedError
        grad_maximum = 1e-5 # the upper bound for the gradient
        intensity_threshold = 0.001 # the lower bound for intensity
        fbp_threshold = 0.05 # threshold for the FBP image
        n_gaussian = 200000 # number of initail gaussians 150K-400K. 
        max_gaussian = 250000
        distance_threshold = 0.1 # threshold for neighbor distance
        radii_coefficient = 0.12 # coefficient for the radii of the gaussians
        intensity_coefficient = 0.3 # coefficient for the intensity of the gaussians
        self.volume_shape = I_FBP.shape
        self.max_gaussian = max_gaussian
        points = torch.rand(n_gaussian, 3).numpy()
        colors = torch.rand(n_gaussian, 1).numpy()
        
        # n_neighbours = count_neighbors(points,distance_threshold)
        n_neighbours = count_neighbors_gpu(points, distance_threshold)
        radii = radii_coefficient / n_neighbours
        colors = intensity_coefficient * colors
        
        colors[colors < intensity_threshold] = intensity_threshold

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        # fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
        fused_color = torch.tensor(np.asarray(colors)).float().cuda()
        features = torch.zeros((fused_color.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :1, 0 ] = fused_color
        features[:, 1:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist = torch.clamp_min(compute_mean_distance(torch.from_numpy(np.asarray(points)).float().cuda()), 1e-7)
        scales = torch.log(torch.sqrt(dist))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_radii2D = torch.from_numpy(radii).cuda()


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    def training_ct_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': 3e-4, "name": "xyz"}, # decay from 2e-4 to 2e-6
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=3e-4,
                                                    lr_final=3e-5,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
    def reset_opacity_ct(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.1))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
 

    def densify_and_split_ct(self, grads, grad_threshold, scene_extent, N=2, increase_factor=1.1):     
        n_init_points = self.get_xyz.shape[0]
        max_gaussian = min(self.max_gaussian, int(n_init_points * increase_factor))
        available_gaussian = max_gaussian - n_init_points
        if available_gaussian <= 0:
            return
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        pts_rank = torch.argsort(torch.norm(padded_grad.unsqueeze(1), dim=-1), descending=True)
        available_gaussian_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool)
        available_gaussian_mask[pts_rank[:available_gaussian]] = True
        selected_pts_mask = torch.logical_and(selected_pts_mask, available_gaussian_mask)
        split_count = torch.sum(selected_pts_mask).item()
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        return split_count
        

    def densify_and_clone_ct(self, grads, grad_threshold, scene_extent, increase_factor=1.1):
        # Extract points that satisfy the gradient condition
        current_gaussian = grads.shape[0]
        max_gaussian = min(self.max_gaussian, int(current_gaussian * increase_factor))
        available_gaussian = max_gaussian - current_gaussian
        if available_gaussian <= 0:
            return
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent) 
          
        pts_rank = torch.argsort(torch.norm(grads, dim=-1), descending=True)
        available_gaussian_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool)
        available_gaussian_mask[pts_rank[:available_gaussian]] = True
        selected_pts_mask = torch.logical_and(selected_pts_mask, available_gaussian_mask)
        clone_count = torch.sum(selected_pts_mask).item()
        indices = torch.where(selected_pts_mask)[0]
        self._features_dc[indices] /= 2
        self._features_rest[indices] /= 2
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask] 
        new_features_rest = self._features_rest[selected_pts_mask] 
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        return clone_count
        
        
    def densify_and_prune_ct(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        prune_mask = torch.where(torch.norm(grads, dim=1) <= 1e-7, True, False)
        # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        # prune_mask = torch.logical_or(prune_mask, big_points_ws)
        prune_count = torch.sum(prune_mask).item()
        self.prune_points(prune_mask)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        clone_count = self.densify_and_clone_ct(grads, max_grad, extent)
        split_count = self.densify_and_split_ct(grads, max_grad, extent)
        torch.cuda.empty_cache()
        print(f"Prune {prune_count} Gaussians, Clone {clone_count} Gaussians, Split {split_count} Gaussians")
    

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1