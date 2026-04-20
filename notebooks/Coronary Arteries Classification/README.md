# Coronary Artery Spatial Classification Pipeline

## 📖 Overview
The `Coronary Classification.ipynb` notebook acts as the anatomical reasoning engine. Rather than relying on a secondary classification neural network, it leverages deterministic spatial geometry and Euclidean distance transforms. It classifies the homogenous arterial tree into three distinct biological entities:
1.  **Right Coronary Artery (RCA)** [Label 1]
2.  **Left Anterior Descending (LAD)** [Label 2]
3.  **Left Circumflex (LCx)** [Label 3]

---

## 🧠 Core Methodology & Pipeline

### Phase 1: Affine Coordination & Optimization
The high-resolution artery arrays and the heart chamber arrays exist in different coordinate spaces. The script uses inverted Affine matrices (`npl.inv(chamber_img.affine)`) to translate arterial voxel coordinates into physical millimeters, and then into the exact voxel space of the TotalSegmentator arrays.
* **Memory Optimization:** Calculating a 3D Euclidean Distance Transform (EDT) on a full 512x512x800 array requires excessive RAM. The script dynamically crops the heart chamber arrays down to the shared bounding box of the heart *and* the mapped skeleton before calculating distances.

### Phase 2: The RCA vs. Left System Split (2-Class)
* **Continuous Branch Logic:** The script uses `scipy.ndimage.label` with a 3x3x3 connected-components structure to group the 1-pixel skeleton into isolated, continuous branches.
* **Majority Voting:** Instead of classifying pixel-by-pixel (which causes messy, alternating colors on a single vessel), the script holds an election for the entire branch.
* **Heuristic:** It calculates the EDT to the Right Atrium (RA) versus the Left Atrium/Ventricle (LA/LV). If a branch has more pixels closer to the RA, the *entire branch* becomes RCA (Label 1). The rest become the Left System (Label 2).

### Phase 3: The Left System Bifurcation Split (LAD vs LCx)
Splitting the left system is biologically complex due to overlapping trajectories.
* **Geometric Biases:**
    * `LCX_BIAS_MM = 5.0`: Artificially penalizes the distance to the Left Atrium, giving the LCx label a slight advantage, accommodating anatomical variance.
    * `LAD_ANCHOR_BIAS_MM = 15.0`: Strongly favors LAD classification for the Left Main Coronary Artery root.
* **Provisional Pixel Classification:** Every pixel is judged. Closer to LA -> LCx. Closer to Anterior wall (RA/RV) -> LAD.
* **Core Extraction & cKDTree Anchoring:** Distal vessel tips often fragment. The script isolates the largest contiguous block of LAD pixels (Main LAD Core) and LCx pixels (Main LCx Core). It builds a K-D Tree (`scipy.spatial.cKDTree`) for both cores. Any floating branches are anchored (reclassified) based on their absolute spatial proximity to these massive core bodies.

### Phase 4: Prioritized 3D Inflation
To prepare the data for ITK-SNAP 3D visualization, the 1-pixel classified trees must be thickened.
* **The Root Overwrite Problem:** Simply dilating the 3 classes simultaneously causes them to overwrite each other at the aortic root, erasing the Left Main or RCA origin.
* **Priority Painting:** The script paints the thick masks in sequence using bitwise logic:
    1.  `thick_canvas[rca_thick] = 1`
    2.  `thick_canvas[lad_thick & ~rca_thick] = 2` (LAD yields to RCA)
    3.  `thick_canvas[lcx_thick & ~rca_thick & ~lad_thick] = 3` (LCx yields to both)

## 📂 Output Specifications
* **Intermediate Thin:** `classified_coronary_tree_<pid>.nii.gz` (2-class, 1px)
* **Intermediate Thick:** `THICK_tree_<pid>.nii.gz` (2-class, 3D)
* **Final Thin:** `full_classified_tree_<pid>.nii.gz` (3-class, 1px)
* **Final Output (Production):** `ICC_Patient_Seg_<pid>.nii.gz` (3-class, thickened, fully classified)