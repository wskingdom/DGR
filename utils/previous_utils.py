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
import pydicom as dicom
import odl
import tqdm
import math
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from odl.discr import grid
from shutil import rmtree

from pydicom.datadict import DicomDictionary #, NameDict, CleanName
from odl.discr.discr_utils import linear_interpolator
from odl.contrib.datasets.ct.mayo_dicom_dict import new_dict_items
from scipy.spatial.transform import Rotation
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
    # if not os.path.exists(save_root):
    #     os.makedirs(save_root)
    if os.path.exists(save_root):
        rmtree(save_root)
    os.makedirs(save_root)
    for i in range(data.shape[2]):
        img = data[:,:,i]
        img = (img - img.min())/(img.max()-img.min()) # range 0-1
        img = (img*255).astype('uint8')
        img = np.rot90(img,1)
        img = img.astype(np.uint8)
        cv2.imwrite(save_root + '/z_%d.png'%i,img)
    for i in range(data.shape[0]):
        img = data[i,:,:]
        img = (img - img.min())/(img.max()-img.min()) # range 0-1
        img = (img*255).astype('uint8')
        img = np.rot90(img,1)
        img = img.astype(np.uint8)
        cv2.imwrite(save_root + '/x_%d.png'%i,img)
    for i in range(data.shape[1]):
        img = data[:,i,:]
        img = (img - img.min())/(img.max()-img.min()) # range 0-1
        img = (img*255).astype('uint8')
        img = np.rot90(img,1)
        img = img.astype(np.uint8)
        cv2.imwrite(save_root + '/y_%d.png'%i,img)
        
        
def save_ct_images2(data,save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    # if os.path.exists(save_root):
    #     rmtree(save_root)
    # os.makedirs(save_root)
    for i in range(data.shape[2]):
        img = data[:,:,i]
        img = (img - img.min())/(img.max()-img.min()) # range 0-1
        img = (img*255).astype('uint8')
        img = np.rot90(img,1)
        img = img.astype(np.uint8)
        cv2.imwrite(os.path.join(save_root,'%03d.png'%i),img)
        np.save(os.path.join(save_root,'%03d.npy'%i),img)


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
    

    volume = np.transpose(np.array(volumes), (1, 2, 0))

    return partition, volume


def dcm2png_mri(dcm_path, save_path):
    image = sitk.ReadImage(dcm_path, sitk.sitkFloat32)

    # Convert the image to a numpy array
    image_array = sitk.GetArrayFromImage(image)

    resp = plt.imsave(save_path, image_array[0], cmap='gray')
    return resp

def dcm2png_ct(dcm_path, save_path):
    image = sitk.ReadImage(dcm_path, sitk.sitkFloat32)
    image_array = sitk.GetArrayFromImage(image)
    image_array = normalise(image_array)
    resp = plt.imsave(save_path, image_array[0], cmap='gray')
    return resp

# def read_image(dcm_path):
#     image = sitk.ReadImage(dcm_path, sitk.sitkFloat32)
#     image_array = sitk.GetArrayFromImage(image)
#     # image_array = normalise(image_array)
#     return image_array[0]


def read_image(dcm_path):
    dataset = dicom.read_file(dcm_path)
    # Get parameters
    pixel_size = np.array(dataset.PixelSpacing)
    pixel_thickness = float(dataset.SliceThickness)
    rows = dataset.Rows
    cols = dataset.Columns

    # Get data array and convert to correct coordinates
    data_array = np.array(np.frombuffer(dataset.PixelData, 'H'),
                            dtype='float32')
    data_array = data_array.reshape([cols, rows], order='C')
    # data_array = np.rot90(data_array, -1)

    # Convert from storage type to densities
    # TODO: Optimize these computations
    hu_values = (dataset.RescaleSlope * data_array +
                    dataset.RescaleIntercept)
    densities = (hu_values + 1000) / 1000
    return densities

def normalise(image):
    # normalise and clip images -1000 to 800
    np_img = image
    np_img = np.clip(np_img, -1000, None).astype(np.float32)
    return np_img


def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret


def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def normalise_one_one(image):
    """Image normalisation. Normalises image to fit [-1, 1] range."""

    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret

if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
