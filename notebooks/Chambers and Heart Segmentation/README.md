# TotalSegmentator Heart Chamber Extraction Pipeline

## 📖 Overview
The `Totalsegmentor_Heart.ipynb` notebook represents the initial step in the coronary artery analysis pipeline. Its primary objective is to extract high-resolution, multi-label heart chambers from raw Cardiac CT scans. This anatomical framework is strictly required for the subsequent spatial classification of coronary arteries.

The script leverages **TotalSegmentator**, a robust deep-learning framework for medical image segmentation, specifically invoking the high-resolution heart chambers model.

---

## ⚙️ Architecture & Dependencies
* **Core Libraries:** `TotalSegmentator`, `nibabel`, `numpy`, `torch`.
* **Hardware Requirements:** GPU execution is mandatory. The script assumes a Colab environment with a Tesla T4 (or equivalent) GPU.
* **License:** Requires a TotalSegmentator academic license key (set internally via `totalseg_set_license`).

---

## 📂 Directory Structure
* **Input Directory:** `/content/drive/MyDrive/COROnet_Drive/ICC_Data` (expects raw CT scans in `.nii.gz` format).
* **Temporary Local Directory:** `/content/temp_out` (Used to bypass Google Drive's slow I/O during active prediction).
* **Final Output Directory:** `/content/drive/MyDrive/COROnet_Drive/Segmentation/Helper/Totalsegmentor_Heart/<patient_id>/`

---

## 🛠️ Step-by-Step Pipeline Explanation

### 1. Environment Initialization
The notebook installs TotalSegmentator and maps the Colab workspace to Google Drive. It registers the required academic license to unlock the V2 models. A crucial utility function, `clear_gpu_memory()`, is defined to invoke Python's garbage collector (`gc.collect()`) and `torch.cuda.empty_cache()` between iterations to prevent VRAM overflow.

### 2. File Discovery & Resume Capability
The script scans the `ICC_Data` directory for `.nii.gz` patient files. It implements a **smart skipping mechanism**: before processing a patient, it checks if `heart.nii.gz` already exists in the destination folder with a valid file size (>1024 bytes). This allows the script to be stopped and safely resumed without redundant processing.

### 3. Inference Execution
For each patient:
1.  **Local Isolation:** The script clears previous temporary files to prevent cross-contamination.
2.  **GPU Prediction:** It executes the TotalSegmentator CLI using the `heartchambers_highres` task (`-ta heartchambers_highres`). It forces GPU usage (`-d gpu`) and enables multi-label output (`--ml`) so all chambers are compiled into a single volume rather than separate binary masks.
3.  **Two-Pass Robustness:** TotalSegmentator inherently performs a fast 3mm rough segmentation for cropping, followed by targeted high-resolution inference on the isolated region.

### 4. Drive Synchronization
Directly writing heavy NIfTI files to Google Drive during inference can cause latency timeouts. The script writes to `/content/temp_out/heart.nii.gz` locally. Once inference is fully complete, Python's `shutil.copy` safely migrates the array to the persistent Drive directory. The local file is deleted post-verification.

---

## 📊 Output Specifications
The primary output is a `heart.nii.gz` file for each patient, mapping the chambers as follows:
* **Label 1:** Left Atrium (LA)
* **Label 2:** Left Ventricle (LV)
* **Label 3:** Right Atrium (RA)
* **Label 4:** Right Ventricle (RV)