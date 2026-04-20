# 🫀 COROnet: Automated Coronary Plaque Classification and Stenosis Grading

**COROnet** is an end-to-end, 3D deep learning diagnostic pipeline designed to automatically analyze Coronary Computed Tomography Angiography (CCTA) scans. It extracts the coronary arterial tree, recovers broken distal branches, and performs high-fidelity atherosclerotic plaque classification and stenosis grading for cardiovascular risk stratification.

### 🌟 Project Overview
The precise characterization of coronary atherosclerosis is critical for predicting adverse cardiovascular events. COROnet addresses the challenges of manual CCTA assessment—such as inter-reader variability and the difficulty of tracing tortuous distal vessels—by combining state-of-the-art volumetric segmentation with a sequence-based diagnostic engine. 

Rather than relying purely on pixel-level segmentation, COROnet treats the artery as a continuous spatial sequence, utilizing 3D patching and Multiple Instance Learning (MIL) to grade lesions with clinical accuracy.

---

### 🧠 Core Architecture & Pipeline

COROnet operates in three distinct phases, bridging anatomical computer vision with diagnostic sequence modeling:

#### Phase 1: High-Fidelity Anatomical Segmentation
* **Heart-ROI Cropping:** Automatically isolates the four cardiac chambers (LA, LV, RA, RV) to remove background noise (lungs, spine) and center the coronary roots.
* **U-Mamba Backbone:** Utilizes a State Space Model (SSM) optimized for long-range 3D dependencies to segment the fine, tortuous structures of the coronary arteries.
* **Anatomical Mapping:** Leverages TotalSegmentator to provide a reliable structural map, assigning definitive labels to the Left Anterior Descending (LAD), Left Circumflex (LCx), and Right Coronary Artery (RCA).

#### Phase 2: Graph-Based Anatomical Healing
Standard neural networks often "break" thin distal vessels. COROnet implements a novel geometric healing algorithm.
* **Skeletonization & Centerline Extraction:** Converts the segmented volumetric masks into a mathematical 3D graph.
* **Aortic Anchoring & Bridging:** Uses KD-Trees and Euclidean Distance Transforms to locate topological gaps and mathematically bridge them with diameter-aware reconstructions, ensuring a continuous path from the aortic ostium to the distal tips.

#### Phase 3: The Diagnostic Engine (3D CNN + BiLSTM + MIL)
* **3D Patch Extraction:** Slides a $32 \times 32 \times 32$ bounding box along the healed centerline, extracting sequential 3D cubes to preserve the true spatial geometry of the vessel wall without interpolation artifacts.
* **Feature Extraction:** A lightweight 3D CNN (e.g., 3D ResNet) compresses each volumetric patch into a rich feature vector.
* **Spatial-Sequential Modeling:** A Bidirectional LSTM (BiLSTM) processes the sequence of patches, understanding the gradual tapering of the artery and the context of blood flow.
* **Attention-Based MIL:** An Attention-Based Multiple Instance Learning layer evaluates the entire vessel as a "bag" of patches, automatically pinpointing the exact locations of lesions to output the final **Stenosis Percentage** and **Plaque Phenotype** (Calcified, Non-Calcified, Mixed).

---

### 📊 Clinical Impact
By decoupling the structural segmentation from the diagnostic classification, COROnet provides a robust, scalable tool for objective CAD diagnosis. It accurately detects lesions even in distal segments, offering clinicians an immediate, standardized assessment of coronary health.