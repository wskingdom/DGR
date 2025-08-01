import os
import numpy as np
import cv2
import tigre
from tigre.utilities import sample_loader
from tqdm import tqdm

def get_geometry(cfg):
    if cfg["mode"] == "parallel":
        geo = tigre.geometry(mode="parallel", nVoxel=np.array(cfg["nVoxel"][::-1]))
    elif cfg["mode"] == "cone":
        geo = tigre.geometry(mode="cone")
    else:
        raise NotImplementedError("Unsupported scanner mode!")

    geo.DSD = cfg["DSD"]  # Distance Source Detector
    geo.DSO = cfg["DSO"]  # Distance Source Origin
    # Detector parameters
    geo.nDetector = np.array(cfg["nDetector"])  # number of pixels
    geo.sDetector = np.array(cfg["sDetector"])  # size of each pixel
    geo.dDetector = geo.sDetector / geo.nDetector  # total size of the detector
    # Image parameters
    geo.nVoxel = np.array(cfg["nVoxel"][::-1])  # number of voxels
    geo.sVoxel = np.array(cfg["sVoxel"][::-1])  # size of each voxel
    geo.dVoxel = geo.sVoxel / geo.nVoxel  # total size of the image
    # Offsets
    geo.offOrigin = np.array(cfg["offOrigin"][::-1])  # Offset of image from origin
    geo.offDetector = np.array(
        [cfg["offDetector"][1], cfg["offDetector"][0], 0]
    )  # Offset of Detector
    # Auxiliary
    geo.accuracy = cfg["accuracy"]  # Accuracy of FWD proj
    # Mode
    geo.filter = cfg["filter"]
    return geo

def init_geometry(cfg_path='/sharedata/datasets/real/real_dataset/pine.txt', n_sample = 50):
    object_scale = 50
    proj_subsample = 4

    with open(cfg_path, "r") as f:
        for config_line in f.readlines():
            if "NumberImages" in config_line:
                n_proj = int(config_line.split("=")[-1])
            elif "AngleInterval" in config_line:
                angle_interval = float(config_line.split("=")[-1])
            elif "AngleFirst" in config_line:
                angle_start = float(config_line.split("=")[-1])
            elif "AngleLast" in config_line:
                angle_last = float(config_line.split("=")[-1])
            elif "DistanceSourceDetector" in config_line:
                DSD = float(config_line.split("=")[-1]) / 1000 * object_scale
            elif "DistanceSourceOrigin" in config_line:
                DSO = float(config_line.split("=")[-1]) / 1000 * object_scale
            elif "PixelSize" in config_line and "PixelSizeUnit" not in config_line:
                dDetector = (
                    float(config_line.split("=")[-1])
                    * proj_subsample
                    / 1000
                    * object_scale
                )
    angles = np.concatenate(
        [np.arange(angle_start, angle_last, angle_interval), [angle_last]]
    )
    angles = angles / 180.0 * np.pi
    # sample angles evenly
    indices = np.linspace(0, len(angles) - 1, n_sample, dtype=int)
    angles = angles[indices]


    nDetector = [512, 512]
    sDetector = np.array(nDetector) * np.array(dDetector)
    nVoxel = [256, 256, 256]
    sVoxel = [2.0, 2.0, 2.0]
    offOrigin = [0.0, 0.0, 0.0]
    scanner_cfg = {
        "mode": "cone",
        "DSD": DSD,
        "DSO": DSO,
        "nDetector": nDetector,
        "sDetector": sDetector.tolist(),
        "nVoxel": nVoxel,
        "sVoxel": sVoxel,
        "offOrigin": offOrigin,
        "offDetector": [0.0, 0.0],
        "accuracy": 0.5,
        "totalAngle": angle_last - angle_start,
        "startAngle": angle_start,
        "noise": True,
        "filter": None,
    }
        

    geo = get_geometry(scanner_cfg)
    return geo, angles

if __name__ == '__main__':
    img_path = '/sharedata/datasets/real/real_dataset/cone_ntrain_25_angle_360/pine/vol_gt.npy'
    pine = np.load(img_path, allow_pickle=True)
    geo, angles = init_geometry()

    for i in tqdm(range(1000)):
        projections = tigre.Ax(pine, geo, angles)

    # 'projections' now contains the 2D projections for each angle
    print("Projection data shape:", projections.shape) 