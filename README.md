# Discretized Gaussian Representation for Tomographic Reconstruction (DGR)

This repository contains the implementation of the proposed DGR algorithm as described in our paper submission.

## File Structure

- **[CT](CT)**: Contains the geometry of the Computed Tomography system.
- **[params](params)**: Contains the parameters (partial) for the DGR algorithm.
- **[utils](utils)**: Contains the utility functions.
- **[DGR.py](DGR.py)**: Contains the implementation of the Discretized Gaussian Representation (DGR).
- **[cone_beam_svct.py](cone_beam_svct.py)**: The reconstruction code for cone beam sparse-view CT.
- **[fan_beam_svct.py](fan_beam_svct.py)**: The reconstruction code for fan beam sparse-view CT.
- **[fan_beam_lact.py](fan_beam_lact.py)**: The reconstruction code for fan beam limited-angle CT.

## Quick Start

Within the comparisons of the 6 Tables in the paper, the cone-beam sparse-view CT is the most cutting-edge task compared to research of SAX-NeRF (CVPR 24), X-Gaussian (ECCV 24), and R$^2$-Gaussian (NeurIPS 24). For a quick start, we advise you to run the cone-beam sparse-view CT with 300 iterations, which will take about 3-4 minutes.

### Step-by-Step Guide

1. **Download the Real-World Dataset**:  
   Please download the real-world dataset from [this website](https://fips.fi/category/open-datasets/x-ray-tomographic-datasets/). The config file also needs to be downloaded.  Following the $R^2$-Gaussian method, you need to reconstruct the pseudo ground truth from these projections.  
   Alternatively, you can directly download the pseudo ground truth provided by the $R^2$-Gaussian team from [Google Drive](https://drive.google.com/drive/folders/1YZ3w87XrCNyjDRos6gkY8zgT5hESl-PN?usp=sharing).

2. **Adjust Dataset Path**:  
   Then, adjust the path of the dataset in the [DGR.py](DGR.py) file:  
   ```python
   self.img_path = 'path/to/your/dataset'
    ```

3. **Adjust Number of Views**:  
   Adjust the number of views in the [DGR.py](DGR.py) file:  
   ```python
   self.n_view = 75/50/25
   ```

4. **Check the path of config file**
  In [cbct_utils.py](cbct_utils.py), you need to adjust the path of the config file.

5. **Run the Code**:  
   Run the code by executing the following command:  
   ```bash
   python cone_beam_svct.py
   ```
6. **Evaluation**:
We provide a quick evaluation during the reconstruction process for PSNR and SSIM. The quick evaluation includes both volume-wise and slice-wise results. Note that the quick evaluation is not the final evaluation. 
In the 6 Tables in the paper, the cone-beam sparse-view CT evaluates with a subset of the 3D volume according to the baseline $R^2$-Gaussian; the fan-beam sparse-view CT evaluates with the full 3D volume according to the baseline SWORD method; the ultra sparse-view CT and limited-angle CT evaluate with axial, coronal, and sagittal slices according to the baseline diffusionMBIR. Please save the reconstructed volume and keep the same format as the baseline method for the final evaluation.

### Change the loss function
In [cone_beam_svct.py](cone_beam_svct.py), [fan_beam_svct.py](fan_beam_svct.py), and [fan_beam_lact.py](fan_beam_lact.py), you can change the loss function by adjusting the following code:
```python
loss = Ll1
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(projs, gaussians.gt_projs.cuda()))
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(projs, gaussians.gt_projs.cuda())) + volume_tv_loss
```


## Note
### Please note that the code is currently in its beta version. We are actively working on it and it will be updated for release soon. 