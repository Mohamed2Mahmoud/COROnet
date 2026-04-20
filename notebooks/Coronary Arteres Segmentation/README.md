# UU-Mamba Coronary Artery Segmentation Pipeline

## 📖 Overview
The `MMBAA_Coronary_Mapping.ipynb` notebook handles the most computationally intensive phase of the project: the extraction of the coronary artery tree from raw CT data. It utilizes a state-of-the-art **UU-Mamba** architecture—a hybrid Mamba (State Space Model) and nnU-Net framework—to capture long-range vascular dependencies efficiently.

---

## ⚙️ Native Environment & Deep Setup
Due to the strict C++ compilation requirements of `causal-conv1d` and `mamba_ssm`, standard Colab/Conda environments often fail. This notebook implements a highly aggressive **Native Python 3.10 deployment**:
1.  Bypasses Colab's default Python environment via `apt-get install python3.10`.
2.  Forces PyTorch `2.2.1` and MONAI `1.3.0` to ensure architectural stability.
3.  Injects pre-compiled `.whl` files for the Dao-AILab Mamba packages.
4.  Modifies nnU-Net's source code dynamically (`predict_from_raw_data.py`) to bypass PyTorch's `weights_only=False` security block that breaks legacy model checkpoints.

---

## 📂 Directory Structure
* **Raw Scans:** `ICC_Data`
* **Heart Masks:** `Segmentation/Helper/COROnet_Heart`
* **Cropped Inputs:** `Segmentation/Cropped_Data`
* **Local Predictions:** `/content/local_preds`
* **Raw Classifier Output:** `Segmentation/Mamba_Coronary_Arteries`
* **Refined Output:** `Segmentation/Coronary_Mapping_Input`

---

## 🛠️ Step-by-Step Pipeline Explanation

### 1. Data Formatting
The script audits the input directory and ensures all `.nii.gz` files conform to the strict nnU-Net naming convention by appending the `_0000` modality suffix.

### 2. Multi-Label Heart Chamber Cropping (Phase 1.5)
Coronary arteries constitute a tiny fraction of a full chest CT. Passing the entire chest through a dense Mamba network would exhaust GPU memory.
* **Mechanism:** Loads the raw CT and the corresponding TotalSegmentator heart mask.
* **Targeting:** Isolates the voxel coordinates where labels 1, 2, 3, or 4 exist (the four chambers).
* **Bounding Box & Padding:** Calculates a 3D bounding box around the heart and applies a **25-voxel spatial padding** to ensure the superficial coronary arteries (which run on the epicardial surface) are not clipped.
* **Execution:** Slices the raw NIfTI array and saves it to `Cropped_Data`.

### 3. UU-Mamba Inference
* **Local Bridge:** Creates symbolic links (`ln -s`) routing Google Drive directories to a local `/workspace/UU-Mamba/data/` structure to trick nnU-Net into running smoothly.
* **Execution:** Calls `nnUNetv2_predict` targeting the `Dataset101_ImageCAS`.
* **Configuration:** Uses the `nnUNetTrainerUMambaEnc` trainer, `3d_fullres` configuration, and explicitly disables Test-Time Augmentation (`--disable_tta`) to vastly improve speed. Runs with 4 parallel workers (`-npp 4 -nps 4`).
* **Sync:** Migrates generated masks back to Drive.

### 4. Post-Processing: Skeletonization & Thickening
Neural network outputs can suffer from varying thickness, uneven boundaries, or micro-fragmentation.
* **Skeletonization:** Reduces the predicted binary mask to a strict 1-voxel centerline using `skimage.morphology.skeletonize`.
* **Uniform Thickening:** Inflates the perfect centerline using a spherical structuring element (`skimage.morphology.ball(6)`), creating uniform vascular tubes with a radius of 6 voxels.
* **Cleanup:** Strips the `_thickened_r6` suffix and routes the perfectly uniform, clean segmented trees to the Classifier Input folder.