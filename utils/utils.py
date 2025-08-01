# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tomographic datasets from Mayo Clinic.

In addition to the standard ODL requirements, this library also requires:

    - tqdm
    - dicom
    - A copy of the Mayo dataset, see
    https://www.aapm.org/GrandChallenge/LowDoseCT/#registration
"""

from __future__ import division
import numpy as np
import os
import sys
import pydicom as dicom
import odl
import tqdm
import math
import cv2
from torch_scatter import scatter_max, scatter_add
from odl.discr import grid
from sklearn.neighbors import KDTree
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pydicom.datadict import DicomDictionary #, NameDict, CleanName
from odl.discr.discr_utils import linear_interpolator
from odl.contrib.datasets.ct.mayo_dicom_dict import new_dict_items
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
# from skimage.metrics import structural_similarity as structural_similarity
# Update the DICOM dictionary with the extra Mayo tags
DicomDictionary.update(new_dict_items)
#NameDict.update((CleanName(tag), tag) for tag in new_dict_items)


__all__ = ('load_projections', 'load_reconstruction')



def convert_to_quaternion(image_orientation_patient):
  """
  Converts image_orientation_patient data (6 elements) to a quaternion.

  Args:
    image_orientation_patient: A list containing 6 direction cosines.

  Returns:
    A list representing the quaternion (w, x, y, z).
  """

  Rx, Ry, Rz, Cx, Cy, Cz = image_orientation_patient

  a = (Rx + Cx) / 2
  b = (Ry + Cy) / 2
  c = (Rz + Cz) / 2
  d = (Rx - Cx) / 2
  e = (Ry - Cy) / 2
  f = (Rz - Cz) / 2

  largest = max(a, b, c, d, e, f)

  if largest == a or largest == b or largest == c:
    w = math.sqrt(1 + a - b - c)
    x = d / (2 * w)
    y = e / (2 * w)
    z = f / (2 * w)
  elif largest == d:
    w = math.sqrt(1 + d - a - b)
    x = 2 * w
    y = e / (2 * w)
    z = f / (2 * w)
  elif largest == e:
    w = math.sqrt(1 + e - a - b)
    x = d / (2 * w)
    y = 2 * w
    z = f / (2 * w)
  else:
    w = math.sqrt(1 + f - a - b)
    x = d / (2 * w)
    y = e / (2 * w)
    z = 2 * w

  return [w, x, y, z]


def transform(image_position_patient, image_orientation_patient, PixelSpacing):
    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[0, 0:3] = image_orientation_patient[0:3] * PixelSpacing[0]
    transformation_matrix[1, 0:3] = image_orientation_patient[3:6] * PixelSpacing[1]
    transformation_matrix[0:3, 3] = image_position_patient
    transformation_matrix[2, 2] = 1
    transformation_matrix[3, 3] = 1
    rotation_matrix = transformation_matrix[0:3, 0:3]
    translation_vector = transformation_matrix[0:3, 3]

    # Convert the rotation matrix to a quaternion
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()

    # Invert the pose
    quaternion = -quaternion
    translation_vector = -translation_vector
    print(image_position_patient, translation_vector)
    return quaternion, translation_vector


def _read_projections_alternative(folder, indices):
    """Read mayo projections from a folder
    following the DICOM-CT-PD User Manual Version 3""" 
    datasets = []

    # Get the relevant file names
    file_names = sorted([f for f in os.listdir(folder) if f.endswith(".dcm")])

    if len(file_names) == 0:
        raise ValueError('No DICOM files found in {}'.format(folder))

    if indices is not None:
        file_names = file_names[indices]

    data_array = None

    for i, file_name in enumerate(tqdm.tqdm(file_names,
                                            'Loading projection data')):
        # read the file
        dataset = dicom.read_file(folder + '/' + file_name)
        dataset.NumberofDetectorRows = dataset[0x70291010].value
        dataset.NumberofDetectorColumns = dataset[0x70291011].value
        dataset.HUCalibrationFactor = float(dataset[0x70411001].value)
        dataset.DetectorFocalCenterAngularPosition = dataset[0x70311001].value
        dataset.DetectorElementTransverseSpacing = dataset[0x70291002].value
        dataset.DetectorElementAxialSpacing = dataset[0x70291006].value
        dataset.DetectorCentralElement = dataset[0x70311033].value
        dataset.DetectorFocalCenterRadialDistance = dataset[0x70311003].value
        dataset.ConstantRadialDistance = dataset[0x70311031].value
        dataset.DetectorFocalCenterAxialPosition = dataset[0x70311002].value
        dataset.SourceAxialPositionShift = dataset[0x7033100C].value # 0x7033100B?
        dataset.SourceAngularPositionShift = dataset[0x7033100B].value
        dataset.SourceRadialDistanceShift = dataset[0x7033100D].value

        # Get some required data
        rows = dataset.NumberofDetectorRows
        cols = dataset.NumberofDetectorColumns
        hu_factor = dataset.HUCalibrationFactor
        rescale_intercept = dataset.RescaleIntercept
        rescale_slope = dataset.RescaleSlope

        # Load the array as bytes
        proj_array = np.array(np.frombuffer(dataset.PixelData, 'H'),
                              dtype='float32')
        proj_array = proj_array.reshape([rows, cols], order='F').T

        # Rescale array
        proj_array *= rescale_slope
        proj_array += rescale_intercept
        proj_array /= hu_factor

        # Store results
        if data_array is None:
            # We need to load the first dataset before we know the shape
            data_array = np.empty((len(file_names), cols, rows),
                                  dtype='float32')

        data_array[i] = proj_array[:, ::-1]
        datasets.append(dataset)

    return datasets, data_array



def load_projections(folder, indices=None):
    """Load geometry and data stored in Mayo format from folder.

    Parameters
    ----------
    folder : str
        Path to the folder where the Mayo DICOM files are stored.
    indices : optional
        Indices of the projections to load.
        Accepts advanced indexing such as slice or list of indices.

    Returns
    -------
    geometry : ConeBeamGeometry
        Geometry corresponding to the Mayo projector.
    proj_data : `numpy.ndarray`
        Projection data, given as the line integral of the linear attenuation
        coefficient (g/cm^3). Its unit is thus g/cm^2.
    """
    datasets, data_array = _read_projections_alternative(folder, indices)

    # Get the angles
    angles = [d.DetectorFocalCenterAngularPosition for d in datasets]
    angles = -np.unwrap(angles) - np.pi  # different definition of angles

    # Set minimum and maximum corners
    shape = np.array([datasets[0].NumberofDetectorColumns,
                      datasets[0].NumberofDetectorRows])
    pixel_size = np.array([datasets[0].DetectorElementTransverseSpacing,
                           datasets[0].DetectorElementAxialSpacing])

    # Correct from center of pixel to corner of pixel
    minp = -(np.array(datasets[0].DetectorCentralElement) - 0.5) * pixel_size
    maxp = minp + shape * pixel_size

    # Select geometry parameters
    src_radius = datasets[0].DetectorFocalCenterRadialDistance
    det_radius = (datasets[0].ConstantRadialDistance -
                  datasets[0].DetectorFocalCenterRadialDistance)

    # For unknown reasons, mayo does not include the tag
    # "TableFeedPerRotation", which is what we want.
    # Instead we manually compute the pitch
    pitch = ((datasets[-1].DetectorFocalCenterAxialPosition -
              datasets[0].DetectorFocalCenterAxialPosition) /
             ((np.max(angles) - np.min(angles)) / (2 * np.pi)))

    # Get flying focal spot data
    offset_axial = np.array([d.SourceAxialPositionShift for d in datasets])
    offset_angular = np.array([d.SourceAngularPositionShift for d in datasets])
    offset_radial = np.array([d.SourceRadialDistanceShift for d in datasets])

    # TODO(adler-j): Implement proper handling of flying focal spot.
    # Currently we do not fully account for it, merely making some "first
    # order corrections" to the detector position and radial offset.

    # Update angles with flying focal spot (in plane direction).
    # This increases the resolution of the reconstructions.
    angles = angles - offset_angular

    # We correct for the mean offset due to the rotated angles, we need to
    # shift the detector.
    offset_detector_by_angles = det_radius * np.mean(offset_angular)
    minp[0] -= offset_detector_by_angles
    maxp[0] -= offset_detector_by_angles

    # We currently apply only the mean of the offsets
    src_radius = src_radius + np.mean(offset_radial)

    # Partially compensate for a movement of the source by moving the object
    # instead. We need to rescale by the magnification to get the correct
    # change in the detector. This approximation is only exactly valid on the
    # axis of rotation.
    mean_offset_along_axis_for_ffz = np.mean(offset_axial) * (
        src_radius / (src_radius + det_radius))

    # Create partition for detector
    detector_partition = odl.uniform_partition(minp, maxp, shape)

    # Convert offset to odl definitions
    offset_along_axis = (mean_offset_along_axis_for_ffz +
                         datasets[0].DetectorFocalCenterAxialPosition -
                         angles[0] / (2 * np.pi) * pitch)

    # Assemble geometry
    angle_partition = odl.nonuniform_partition(angles)
    geometry = odl.tomo.ConeBeamGeometry(angle_partition,
                                         detector_partition,
                                         src_radius=src_radius,
                                         det_radius=det_radius,
                                         pitch=pitch,
                                         offset_along_axis=offset_along_axis)

    # Create a *temporary* ray transform (we need its range)
    spc = odl.uniform_discr([-1] * 3, [1] * 3, [32] * 3)
    ray_trafo = odl.tomo.RayTransform(spc, geometry, interp='linear')

    # convert coordinates
    theta, up, vp = ray_trafo.range.grid.meshgrid
    d = src_radius + det_radius
    u = d * np.arctan(up / d)
    v = d / np.sqrt(d**2 + up**2) * vp

    # Calculate projection data in rectangular coordinates since we have no
    # backend that supports cylindrical
    interpolator = linear_interpolator(
        data_array, ray_trafo.range.grid.coord_vectors
    )
    proj_data = interpolator((theta, u, v))

    return geometry, proj_data#.asarray()


def save_ct_images(data,save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for i in range(data.shape[2]):
        img = data[:,:,i]
        img = (img - img.min())/(img.max()-img.min()) # range 0-1
        img = (img*255).astype('uint8')
        img = np.rot90(img,1)
        img = img.astype(np.uint8)
        cv2.imwrite(save_root + '/%d.png'%i,img)


def load_npy(root):
    fname_list = os.listdir(root)
    fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
    print(fname_list)
    all_img = []

    print("Loading all data")
    for fname in tqdm.tqdm(fname_list):
        just_name = fname.split('.')[0]
        img = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))
        h, w = img.shape
        # img = img.view(1, 1, h, w)
        img = img.view(1, h, w)
        all_img.append(img)
    all_img = torch.cat(all_img, dim=0)
    print(f"Data loaded shape : {all_img.shape}")
    img = all_img.cuda()
    return img


def load_reconstruction(folder, slice_start=0, slice_end=-1, suffix='.IMA'):
    """Load a volume from folder, also returns the corresponding partition.

    Parameters
    ----------
    folder : str
        Path to the folder where the DICOM files are stored.
    slice_start : int
        Index of the first slice to use. Used for subsampling.
    slice_end : int
        Index of the final slice to use.

    Returns
    -------
    partition : `odl.RectPartition`
        Partition describing the geometric positioning of the voxels.
    data : `numpy.ndarray`
        Volumetric data. Scaled such that data = 1 for water (0 HU).

    Notes
    -----
    DICOM data is highly non trivial. Typically, each slice has been computed
    with a slice tickness (e.g. 3mm) but the slice spacing might be
    different from that.

    Further, the coordinates in DICOM is typically the *middle* of the pixel,
    not the corners as in ODL.

    This function should handle all of these peculiarities and give a volume
    with the correct coordinate system attached.
    """
    file_names = sorted([f for f in os.listdir(folder) if f.endswith(suffix)])

    if len(file_names) == 0:
        raise ValueError('No DICOM files found in {}'.format(folder))

    volumes = []
    datasets = []

    # file_names = file_names[slice_start:slice_end]

    for file_name in tqdm.tqdm(file_names, 'loading volume data'):
        # read the file
        dataset = dicom.read_file(folder + '/' + file_name)

        # Get parameters
        pixel_size = np.array(dataset.PixelSpacing)
        pixel_thickness = float(dataset.SliceThickness)
        rows = dataset.Rows
        cols = dataset.Columns

        # Get data array and convert to correct coordinates
        data_array = np.array(np.frombuffer(dataset.PixelData, 'H'),
                              dtype='float32')
        data_array = data_array.reshape([cols, rows], order='C')
        data_array = np.rot90(data_array, -1)

        # Convert from storage type to densities
        # TODO: Optimize these computations
        hu_values = (dataset.RescaleSlope * data_array +
                     dataset.RescaleIntercept)
        densities = (hu_values + 1000) / 1000

        # Store results
        volumes.append(densities)
        datasets.append(dataset)
        # pose = transform(np.array(dataset.ImagePositionPatient), np.array(dataset.ImageOrientationPatient),np.array(dataset.PixelSpacing))

    voxel_size = np.array(list(pixel_size) + [pixel_thickness])
    shape = np.array([rows, cols, len(volumes)])

    # Compute geometry parameters
    mid_pt = (np.array(dataset.ReconstructionTargetCenterPatient) -
              np.array(dataset.DataCollectionCenterPatient))
    reconstruction_size = (voxel_size * shape)
    min_pt = mid_pt - reconstruction_size / 2
    max_pt = mid_pt + reconstruction_size / 2

    # axis 1 has reversed convention
    min_pt[1], max_pt[1] = -max_pt[1], -min_pt[1] # !!!!!!!!!!!!!!!!!!!!!! check this line 2024.3.9

    if len(datasets) > 1:
        slice_distance = np.abs(
            float(datasets[1].DataCollectionCenterPatient[2]) -
            float(datasets[0].DataCollectionCenterPatient[2]))
    else:
        # If we only have one slice, we must approximate the distance.
        slice_distance = pixel_thickness

    # The middle of the minimum/maximum slice can be computed from the
    # DICOM attribute "DataCollectionCenterPatient". Since ODL uses corner
    # points (e.g. edge of volume) we need to add half a voxel thickness to
    # both sides.
    try:
        min_pt[2] = -np.array(datasets[0].DataCollectionCenterPatient)[2]
        min_pt[2] -= 0.5 * slice_distance
        max_pt[2] = -np.array(datasets[-1].DataCollectionCenterPatient)[2]
        max_pt[2] += 0.5 * slice_distance
        partition = odl.uniform_partition(min_pt, max_pt, shape)
    except:
        print('!!!!!!!!Load data in the reverse order!!!!!!!!!!')
        min_pt[2] = -np.array(datasets[-1].DataCollectionCenterPatient)[2]
        min_pt[2] -= 0.5 * slice_distance
        max_pt[2] = -np.array(datasets[0].DataCollectionCenterPatient)[2]
        max_pt[2] += 0.5 * slice_distance
        partition = odl.uniform_partition(min_pt, max_pt, shape)
    

    # volume = np.transpose(np.array(volumes), (1, 2, 0))
    volume = np.array(volumes)
    return partition, volume


def resize_and_crop(image, image_size):
        # Crop slices in z dim
    center_idx = int(image.shape[0] / 2)
    num_slice = int(image_size[0] / 2)
    image = image[center_idx-num_slice:center_idx+num_slice, :, :]
    im_size = image.shape
    print(image.shape, center_idx, num_slice)

    # Complete 3D input image as a squared x-y image
    if not(im_size[1] == im_size[2]):
        length = np.max([im_size[1], im_size[2]])
        zerp_padding = np.zeros([im_size[0], length, length])
        if im_size[1] > im_size[2]:
            zerp_padding[:, :, np.int64((im_size[1]-im_size[2])/2):np.int64((im_size[1]-im_size[2])/2)+im_size[2]] = image
        else:
            zerp_padding[:, np.int64((im_size[2]-im_size[1])/2):np.int64((im_size[2]-im_size[1])/2)+im_size[1], :] = image
        image = zerp_padding
    # Resize image in x-y plane
    image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
    image = F.interpolate(image, size=(image_size[1], image_size[2]), mode='bilinear', align_corners=False)

    # Scaling normalization
    image = image / torch.max(image)  # [B, C, H, W], [0, 1]
    # image = (image-image.min())/(image.max()-image.min())
    image = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
    return image


def resize_and_crop_sr(image, image_size):
        # Crop slices in z dim
    center_idx = int(image.shape[0] / 2)
    num_slice = int(image_size[0] / 2)
    image = image[center_idx-num_slice:center_idx+num_slice, :, :]
    im_size = image.shape
    print(image.shape, center_idx, num_slice)

    # Complete 3D input image as a squared x-y image
    if not(im_size[1] == im_size[2]):
        length = np.max([im_size[1], im_size[2]])
        zerp_padding = np.zeros([im_size[0], length, length])
        if im_size[1] > im_size[2]:
            zerp_padding[:, :, np.int64((im_size[1]-im_size[2])/2):np.int64((im_size[1]-im_size[2])/2)+im_size[2]] = image
        else:
            zerp_padding[:, np.int64((im_size[2]-im_size[1])/2):np.int64((im_size[2]-im_size[1])/2)+im_size[1], :] = image
        image = zerp_padding
    # Resize image in x-y plane
    image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
    image_lr = F.interpolate(image, size=(image_size[1], image_size[2]), mode='bilinear', align_corners=False)

    # Scaling normalization
    image = image / torch.max(image)  # [B, C, H, W], [0, 1]
    # image = (image-image.min())/(image.max()-image.min())
    image = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
    image_lr = image_lr / torch.max(image_lr)  # [B, C, H, W], [0, 1]
    image_lr = image_lr.permute(1, 2, 3, 0)  # [C, H, W, 1]
    return image, image_lr

def fast_volume_reconstruction(gaussians, target_volume_shape, size=[11, 11, 11]):
    gaussian_features = gaussians.get_features.transpose(1, 2).view(-1, 1, (gaussians.max_sh_degree+1)**2)
    colors = gaussian_features[:, :1, 0] # * gaussians.get_opacity
    colors = torch.clamp_min(colors, 0.0)
    coords = gaussians.get_xyz
    symm, actual_covariances, covariances, std = gaussians.get_covariance(1.0)
    n_gaussians = coords.shape[0]
    # extent = 0.05
    target_volume = torch.zeros(target_volume_shape).cuda().reshape(-1)
    impact_shape = torch.tensor(size).int().cuda()
    ranges = (impact_shape / torch.tensor(target_volume_shape).cuda()) / 2
    grid_i, grid_j, grid_k = torch.meshgrid(
        torch.linspace(-ranges[0], ranges[0], impact_shape[0]),
        torch.linspace(-ranges[1], ranges[1], impact_shape[1]),
        torch.linspace(-ranges[2], ranges[2], impact_shape[2]),
    )
    impact_coords = torch.stack([grid_i, grid_j, grid_k], dim=-1).cuda() # 13, 13, 5, 3
    coords = coords * torch.tensor(target_volume_shape).cuda()
    delta_impact_coords = (coords - coords.floor()) / torch.tensor(target_volume_shape).cuda()
    coords = coords.floor().long()
    contributions = colors.reshape(-1,1,1,1) * torch.exp(-0.5 * decomposed_gaussian_computation(impact_coords, delta_impact_coords, torch.inverse(covariances)))
    shifts = torch.meshgrid(
        torch.arange(-(impact_shape[0]//2), impact_shape[0]//2 + 1),
        torch.arange(-(impact_shape[1]//2), impact_shape[1]//2 + 1),
        torch.arange(-(impact_shape[2]//2), impact_shape[2]//2 + 1),
    )
    shifts = torch.stack(shifts, dim=-1).cuda()
    coords = coords[:, None, None, None, :] + shifts
    invalid_indices = (coords[..., 0] < 0) | (coords[..., 0] >= target_volume_shape[0]) | (coords[..., 1] < 0) | (coords[..., 1] >= target_volume_shape[1]) | (coords[..., 2] < 0) | (coords[..., 2] >= target_volume_shape[2])
    positions = coords[..., 0] * target_volume_shape[1] *target_volume_shape[2] + coords[..., 1] * target_volume_shape[2] + coords[..., 2]
    contributions = contributions.reshape(-1)
    positions = positions.reshape(-1)
    valid_indices = (~invalid_indices).reshape(-1)
    target_volume.scatter_add_(0, positions[valid_indices], contributions[valid_indices])
    # volume, _ = scatter_max(contributions[valid_indices], positions[valid_indices], dim=0)
    # target_volume = torch.zeros(target_volume_shape).reshape(-1).cuda()
    # target_volume[:volume.shape[0]] = volume
    target_volume = target_volume.reshape(target_volume_shape)
    return target_volume, impact_shape, std

def resize_volume(volume, size=(256, 256), save_root=None):
    volume = F.interpolate(torch.tensor(volume, dtype=torch.float32)[None, ...], size=size, mode='bilinear', align_corners=False)[0]
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    volume = np.rot90(volume.numpy(), k=1, axes=(1, 2))
    if save_root is not None:
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for i in range(volume.shape[0]):
            image = volume[i]
            cv2.imwrite(os.path.join(save_root, f'{i:04d}.png'), (image*255).astype(np.uint8))
    return volume

def resize_volume2(volume, size=(256, 256), save_root=None):
    volume = torch.tensor(volume, dtype=torch.float32)[None, ...].cuda()
    volume = F.interpolate(volume, size=size, mode='bilinear', align_corners=False)[0]
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for i in range(volume.shape[0]):
        cv2.imwrite(os.path.join(save_root, f'{i:04d}.png'), (volume[i]).cpu().numpy().astype(np.uint8))
    return volume


def evaluation_3d(recon_volume,gt_volume):
    recon_volume = recon_volume.squeeze().cpu().numpy()
    gt_volume = gt_volume.squeeze().cpu().numpy()
    psnr, ssim = [], []
    for i in range(recon_volume.shape[2]):
        recon_axial = recon_volume[:,:,i]
        gt_axial = gt_volume[:,:,i]
        psnr_axial, ssim_axial = peak_signal_noise_ratio(recon_axial, gt_axial, data_range=1), structural_similarity(recon_axial, gt_axial, multichannel=False, data_range=1)
        psnr.append(psnr_axial)
        ssim.append(ssim_axial)
    print('axial PSNR:', np.mean(psnr))
    print('axial SSIM:', np.mean(ssim))

    # compute coronal view
    psnr, ssim = [], []
    for i in range(recon_volume.shape[0]):
        recon_coronal = recon_volume[i,:,:]
        gt_coronal = gt_volume[i,:,:]
        psnr_coronal, ssim_coronal = peak_signal_noise_ratio(recon_coronal, gt_coronal, data_range=1), structural_similarity(recon_coronal, gt_coronal, multichannel=False, data_range=1)
        psnr.append(psnr_coronal)
        ssim.append(ssim_coronal)
    print('coronal PSNR:', np.mean(psnr))
    print('coronal SSIM:', np.mean(ssim))

    # compute sagittal view
    psnr, ssim = [], []
    for i in range(recon_volume.shape[1]):
        recon_sagittal = recon_volume[:,i,:]
        gt_sagittal = gt_volume[:,i,:]
        psnr_sagittal, ssim_sagittal = peak_signal_noise_ratio(recon_sagittal, gt_sagittal, data_range=1), structural_similarity(recon_sagittal, gt_sagittal, multichannel=False, data_range=1)
        psnr.append(psnr_sagittal)
        ssim.append(ssim_sagittal)
    print('sagittal PSNR:', np.mean(psnr))
    print('sagittal SSIM:', np.mean(ssim))

    sys.stdout.flush()

def quick_evaluation(test_output,test_data):
    mse = torch.mean((test_output - test_data)**2)
    # max_pixel_value = torch.max(test_data)
    # if max_pixel_value != 1:
    #     print('max_pixel_value:', max_pixel_value)
    test_psnr = 10 * torch.log10(1 / mse)
    test_ssim = structural_similarity(test_output.squeeze().cpu().numpy(), test_data.squeeze().cpu().numpy(), multichannel=True, data_range=1)
    print('MSE:{:.4f}, PSNR: {:.4f}, SSIM: {:.4f}'.format(mse*255*255/1000, test_psnr, test_ssim))
    sys.stdout.flush()

# def quick_evaluation(test_output,test_data):
#     loss_fn = torch.nn.MSELoss()
#     test_loss = 0.5 * loss_fn(test_output, test_data)
#     test_psnr = - 10 * torch.log10(2 * test_loss).item()
#     test_loss = test_loss.item()
#     test_ssim = structural_similarity(test_output.squeeze().cpu().numpy(), test_data.squeeze().cpu().numpy(), multichannel=True, data_range=2)
#     print('Test Loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}'.format(test_loss, test_psnr, test_ssim))

def count_neighbors(points, distance_threshold):
    tree = KDTree(points)
    neighbor_count = np.zeros(points.shape[0])
    for i in tqdm.tqdm(range(len(points))):
        num_neighbors = tree.query_radius(points[i].reshape(1,-1), distance_threshold, count_only=True)[0]
        neighbor_count[i] = num_neighbors - 1
    return neighbor_count


def compute_mean_distance(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

def count_neighbors_gpu(points, distance_threshold, batch_size=4096):
    points = torch.tensor(points).cuda()  # Move data to GPU
    distance_threshold = torch.tensor(distance_threshold).cuda()

    num_points = points.shape[0]
    neighbor_count = torch.zeros(num_points, device='cuda')

    # Process each batch separately
    for i in tqdm.tqdm(range(0, num_points, batch_size)):
        batch_points = points[i:i+batch_size]
        dist_matrix = torch.cdist(batch_points, points)  # Compute distances to all points
        neighbor_count[i:i+batch_size] = torch.sum(dist_matrix < distance_threshold, dim=1) - 1  # Count neighbors for each point in batch

    return neighbor_count.cpu().numpy()  # Move data back to CPU


def patched_fast_gaussian_computation(A, B, C, patch_size = 10000):
    _B = B[:,None,None,:]
    ATCA = []
    ATCB = []
    BTCA = []
    for i in range(0, B.shape[0], patch_size):
        ATCA.append(torch.einsum('ijkl,bll,ijkl->bijk',A,C[i: i+patch_size],A))
    ATCA = torch.cat(ATCA, dim=0)
    # del ATCA
    for i in range(0, B.shape[0], patch_size):
        ATCB.append(torch.einsum('ijkl,bll,bxxl->bijk',A,C[i: i+patch_size],_B[i:i+patch_size]))
    ATCB = torch.cat(ATCB, dim=0)
    # del ATCB
    for i in range(0, B.shape[0], patch_size):
        BTCA.append(torch.einsum('bxxl,bll,ijkl->bijk',_B[i: i+patch_size],C[i: i+patch_size],A))
    BTCA = torch.cat(BTCA, dim=0)
    # del BTCA
    BTCB =  torch.einsum('bxxl,bll,bxxl->b',_B,C,_B)[:,None,None,None]
    return ATCA - ATCB - BTCA + BTCB


def decomposed_gaussian_computation(A, B, C):
    _B = B[:,None,None,:]
    ATCA = torch.einsum('ijkl,bll,ijkl->bijk',A,C,A)
    ATCB = torch.einsum('ijkl,bll,bxxl->bijk',A,C,_B)
    BTCA = torch.einsum('bxxl,bll,ijkl->bijk',_B,C,A)
    BTCB = torch.einsum('bxxl,bll,bxxl->b',_B,C,_B)
    return ATCA - ATCB - BTCA + BTCB[:,None,None,None]



def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False):
  mux_in = size ** 2
  if type.endswith('2d'):
    Nsamp = mux_in // acc_factor
  elif type.endswith('1d'):
    Nsamp = size // acc_factor
  if type == 'gaussian2d':
    mask = torch.zeros_like(img)
    cov_factor = size * (1.5 / 128)
    mean = [size // 2, size // 2]
    cov = [[size * cov_factor, 0], [0, size * cov_factor]]
    if fix:
      samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
      int_samples = samples.astype(int)
      int_samples = np.clip(int_samples, 0, size - 1)
      mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
    else:
      for i in range(batch_size):
        # sample different masks for batch
        samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
  elif type == 'uniformrandom2d':
    mask = torch.zeros_like(img)
    if fix:
      mask_vec = torch.zeros([1, size * size])
      samples = np.random.choice(size * size, int(Nsamp))
      mask_vec[:, samples] = 1
      mask_b = mask_vec.view(size, size)
      mask[:, ...] = mask_b
    else:
      for i in range(batch_size):
        # sample different masks for batch
        mask_vec = torch.zeros([1, size * size])
        samples = np.random.choice(size * size, int(Nsamp))
        mask_vec[:, samples] = 1
        mask_b = mask_vec.view(size, size)
        mask[i, ...] = mask_b
  elif type == 'gaussian1d':
    mask = torch.zeros_like(img)
    mean = size // 2
    std = size * (15.0 / 128)
    Nsamp_center = int(size * center_fraction)
    if fix:
      samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
      int_samples = samples.astype(int)
      int_samples = np.clip(int_samples, 0, size - 1)
      mask[... , int_samples] = 1
      c_from = size // 2 - Nsamp_center // 2
      mask[... , c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp*1.2))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[i, :, :, int_samples] = 1
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from + Nsamp_center] = 1
  elif type == 'uniform1d':
    mask = torch.zeros_like(img)
    if fix:
      Nsamp_center = int(size * center_fraction)
      samples = np.random.choice(size, int(Nsamp - Nsamp_center))
      mask[..., samples] = 1
      # ACS region
      c_from = size // 2 - Nsamp_center // 2
      mask[..., c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        Nsamp_center = int(size * center_fraction)
        samples = np.random.choice(size, int(Nsamp - Nsamp_center))
        mask[i, :, :, samples] = 1
        # ACS region
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from+Nsamp_center] = 1
  else:
    NotImplementedError(f'Mask type {type} is currently not supported.')

  return mask


def total_variation_loss(t):
    loss = torch.mean(torch.abs(t[:, :, :-1] - t[:, :, 1:])) + torch.mean(torch.abs(t[:, :-1, :] - t[:, 1:, :])) + torch.mean(torch.abs(t[:-1, :, :] - t[1:, :, :]))
    return loss


def init_ct_op(img,r, Fan_ray_trafo, Fan_FBP):
    img = img.cpu().numpy()
    batch = img.shape[0]
    sinogram = np.zeros([batch,720,720])

    sparse_sinogram = np.zeros([batch,720,720])
    ori_img = np.zeros_like(img)
    sinogram_max = np.zeros([batch,1])
    for i in range(batch):
        sinogram[i,...] = Fan_ray_trafo(img[i,...]).data
        ori_img[i,...] = Fan_FBP(sinogram[i,...]).data
        sinogram_max[i,0] = sinogram[i,...].max()
        t = np.copy(sinogram[i,::r,:])
        sparse_sinogram[i,...] = cv2.resize(t,[720,720])
    
    return ori_img, sparse_sinogram.astype(np.float32), sinogram.astype(np.float32),sinogram_max





if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
