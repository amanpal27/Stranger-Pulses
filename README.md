# Stranger Pulses: HTRU2 Pulsar Detection & Analysis
### Astroclub Winter Project 2025-26

Stranger Pulses is an astrophysics and machine learning project focused on detecting **Pulsars** (highly magnetized, rapidly rotating neutron stars) using the HTRU2 dataset from the High Time Resolution Universe Survey. The project implements a complete data pipeline that handles extreme class imbalances (real pulsars represent only ~9% of candidates) to train and compare seven classification algorithms.

---

## Table of Contents
1. [Astrophysics Theoretical Foundation](#astrophysics-theoretical-foundation)
   - [Life Cycle of Stars & The HR Diagram](#life-cycle-of-stars--the-hr-diagram)
   - [Pulsars and Their Discovery](#pulsars-and-their-discovery)
   - [Signal Propagation & Dispersion Measure (DM)](#signal-propagation--dispersion-measure-dm)
   - [De-dispersion, Fourier Transforms, and Folding](#de-dispersion-fourier-transforms-and-folding)
   - [Magnetic Dipole Radiation & Spin-Down](#magnetic-dipole-radiation--spin-down)
2. [Machine Learning Pipeline](#machine-learning-pipeline)
   - [Dataset Overview (HTRU2)](#dataset-overview-htru2)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Data Preprocessing & SMOTE](#data-preprocessing--smote)
   - [Dimensionality Reduction (PCA)](#dimensionality-reduction-pca)
3. [Model Training & Evaluation](#model-training--evaluation)
   - [Performance Comparison](#performance-comparison)
   - [Key Insights & Best Performing Model](#key-insights--best-performing-model)
4. [Repository Structure](#repository-structure)
5. [Installation & Usage](#installation--usage)

---

## Astrophysics Theoretical Foundation

### Life Cycle of Stars & The HR Diagram
Stars evolve through distinct phases depending on their initial mass. This lifecycle is classified on the **Hertzsprung-Russell (H-R) Diagram**, which plots stellar surface temperature (x-axis, decreasing left-to-right) against luminosity (y-axis).
* **Main Sequence**: Stable stars (like our Sun) that burn hydrogen into helium in their cores.
* **White Dwarfs**: Remnants of low-to-medium mass stars ($M < 8 M_\odot$). Squeezed into an earth-sized sphere with densities around $10^6 \text{ g/cm}^3$, they are stabilized against gravity by **electron degeneracy pressure**.
* **Neutron Stars (including Pulsars)**: Formed when massive stars ($8 M_\odot < M < 25 M_\odot$) undergo supernova explosions. Their cores collapse until protons and electrons merge into neutrons, creating objects with atomic-nucleus density ($\sim 10^{14} \text{ g/cm}^3$) stabilized by **neutron degeneracy pressure**.
* **Black Holes**: Formed by the core collapse of extremely massive stars ($M > 25 M_\odot$) where gravity overwhelms degeneracy pressure, collapsing the core into a singularity.

### Pulsars and Their Discovery
Pulsars are rapidly rotating, highly magnetized neutron stars. They act as cosmic lighthouses, emitting narrow beams of electromagnetic radiation from their magnetic poles. As the star rotates, these beams sweep across Earth's line of sight, creating highly periodic pulses.

* **Discovery**: First discovered in 1967 by Jocelyn Bell Burnell and Antony Hewish. They detected highly regular radio pulses repeating every 1.337 seconds, initially designated **LGM-1** ("Little Green Men-1") before being confirmed as CP1919.
* **The Density Condition**: The rapid spin period ($P$) restricts the star's minimum density. If a star rotates too fast, centrifugal forces will rip it apart unless gravity holds it together:
  $$\rho \ge \frac{3\pi}{G P^2}$$
  For a period of $\sim 1 \text{ second}$, the required density is $\approx 10^8 \text{ g/cm}^3$, which exceeds the limit of stable white dwarfs, proving that pulsars must be neutron stars.

### Signal Propagation & Dispersion Measure (DM)
As pulsar signals travel through the interstellar medium (ISM), they interact with free electrons in the ionized plasma. This causes dispersion: lower-frequency radio waves travel slower than higher-frequency ones.
The dispersion delay $\Delta t$ between two frequencies $f_1$ and $f_2$ (with $f_1 > f_2$) is:
$$\Delta t \approx 4.15 \times 10^3 \left( f_2^{-2} - f_1^{-2} \right) \times \text{DM} \text{ seconds}$$

Where the **Dispersion Measure (DM)** is the total column density of free electrons ($n_e$) along the path to the pulsar:
$$\text{DM} = \int_{0}^{d} n_e \, dl \quad [\text{pc cm}^{-3}]$$

### De-dispersion, Fourier Transforms, and Folding
Because pulsar signals are extremely weak and dispersed by the ISM, they arrive at radio telescopes smeared in time. Identifying them requires signal processing:
1. **De-dispersion**: Telemetry is divided into frequency channels, and the frequency-dependent time delays are shifted backwards to align the pulses back into a single, coherent spike.
2. **Fourier Transform**: Raw time-series observations are converted into the frequency domain. Buried periodic pulses appear as sharp spikes at the pulsar’s fundamental rotation frequency ($f = 1/P$) and its harmonics.
3. **Folding**: The time-series data is sliced into blocks equal to the suspected period $P$ and stacked. While random noise cancels out (scaling as $\sqrt{N}$), the coherent pulse signal grows linearly with the number of folds ($N$), dramatically increasing the Signal-to-Noise Ratio (SNR).

### Magnetic Dipole Radiation & Spin-Down
Pulsars possess intense magnetic fields ($10^8$ to $10^{12}$ Gauss). If the magnetic dipole axis is tilted at an angle $\alpha$ relative to the rotation axis, the rotating dipole radiates energy. The energy loss rate (spin-down luminosity $L$) is:
$$L = - \frac{dE_{\text{rot}}}{dt} = I \Omega \dot{\Omega} \propto \Omega^4 B^2 R^6 \sin^2\alpha$$

This constant emission of electromagnetic dipole radiation drains the star's rotational kinetic energy, causing the pulsar's rotation period to slowly increase over time (known as **spin-down**).

---

## Machine Learning Pipeline

### Dataset Overview (HTRU2)
Pulsar candidates are evaluated using the HTRU2 dataset from the High Time Resolution Universe Survey. The dataset contains **17,898 candidate profiles**:
* **Class 0 (Noise / RFI)**: 16,259 profiles (~90.8%)
* **Class 1 (Pulsar)**: 1,639 profiles (~9.2%)

Each candidate is described by 8 continuous statistical features:
1. **Mean** of the integrated profile.
2. **Standard deviation** of the integrated profile.
3. **Excess kurtosis** of the integrated profile.
4. **Skewness** of the integrated profile.
5. **Mean** of the DM-SNR curve.
6. **Standard deviation** of the DM-SNR curve.
7. **Excess kurtosis** of the DM-SNR curve.
8. **Skewness** of the DM-SNR curve.

### Exploratory Data Analysis (EDA)
EDA shows distinct differences between noise and pulsar signals:
* **Feature Correlation**: The integrated profile's kurtosis and skewness exhibit high positive correlation ($r \approx 0.94$). Real pulsars tend to have sharp, asymmetric peaks (higher excess kurtosis and skewness) compared to broader, symmetric RFI.
* **Class Separation**: Visual plots reveal that noise candidates crowd in different sub-regions of the feature space, but overlap significantly near transition boundaries, necessitating non-linear classification models.

### Data Preprocessing & SMOTE
1. **Train-Test Split**: The dataset is split into 75% training and 25% testing datasets using stratified splitting to preserve the ~9% minority class ratio in both sets.
2. **Feature Scaling**: Due to varying feature magnitudes (e.g. mean profile vs. DM-SNR kurtosis), data is normalized using `StandardScaler` to ensure distance-based models (KNN, SVM) are unbiased.
3. **SMOTE (Synthetic Minority Over-sampling Technique)**: To prevent classifiers from prioritizing the majority class, SMOTE is applied **exclusively to the training set** to synthesize new minority class (pulsar) examples by interpolating between nearest neighbors. Balancing the train data prevents data leakage while maximizing recall.

### Dimensionality Reduction (PCA)
Principal Component Analysis (PCA) is applied to the scaled features to project the 8D space into lower-dimensional spaces (1D, 2D, and 3D) for visualization. 
* The first three Principal Components (PCs) account for a high percentage of the total variance, showing a clear spatial division where pulsar candidates cluster together in a distinct branch, separate from the core RFI noise cluster.

---

## Model Training & Evaluation

Seven classifiers were trained and tested on both the **Normal (Imbalanced)** and **SMOTE-Balanced** datasets:
* Logistic Regression
* Random Forest (1000 estimators)
* Naive Bayes (Gaussian)
* K-Nearest Neighbors (KNN)
* Decision Tree
* MLP Neural Network
* Support Vector Machine (SVM)

### Performance Comparison

| Model | Dataset | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Normal | 98.01% | 94.04% | 83.41% | 88.41% | 97.94% |
| **SVM** | Normal | 97.97% | 94.81% | 82.20% | 88.05% | 96.95% |
| **MLP (Neural Net)** | Normal | 97.99% | 93.30% | 84.39% | 88.62% | 97.92% |
| **Logistic Regression**| Normal | 97.88% | 93.44% | 82.68% | 87.73% | 97.74% |
| **KNN** | Normal | 97.90% | 92.51% | 83.90% | 88.00% | 95.84% |
| **Decision Tree** | Normal | 96.79% | 82.78% | 82.20% | 82.49% | 90.23% |
| **Naive Bayes** | Normal | 94.46% | 66.86% | 79.51% | 72.64% | 96.34% |
| **Random Forest** | SMOTE | 97.66% | 85.34% | 89.76% | 87.49% | 97.73% |
| **SVM** | SMOTE | 97.30% | 80.95% | 92.20% | 86.21% | 97.80% |
| **MLP (Neural Net)** | SMOTE | 97.41% | 81.99% | 91.95% | 86.69% | 97.91% |
| **Logistic Regression**| SMOTE | 96.65% | 76.54% | 91.71% | 83.44% | 97.80% |
| **KNN** | SMOTE | 96.47% | 75.31% | 91.71% | 82.70% | 95.66% |
| **Decision Tree** | SMOTE | 96.47% | 79.03% | 83.41% | 81.16% | 90.60% |
| **Naive Bayes** | SMOTE | 89.25% | 45.45% | 87.80% | 59.85% | 95.85% |

### Key Insights & Best Performing Model
* **The Imbalance Challenge**: Models trained on the imbalanced (Normal) dataset exhibit very high Precision but suffer from lower Recall ($\approx 82\text{--}84\%$), meaning they fail to classify roughly $16\text{--}18\%$ of actual pulsars.
* **The SMOTE Benefit**: Training with SMOTE data increases the **Recall** of the models (up to **92.2%** for SVM and **91.9%** for MLP) by allowing them to recognize more subtle pulsar structures, though this leads to a minor drop in Precision due to increased false positives.
* **Best Classifier**:
  - The **MLP (Neural Net)** model and **Random Forest** achieve the highest overall balance of F1-Score and AUC.
  - When raw detection coverage is preferred (prioritizing finding every candidate pulsar), the **SMOTE-trained SVM** or **MLP** is optimal due to their high recall ($>91.9\%$).

---

## Repository Structure

```
Stranger-Pulses/
├── .gitignore             # Git ignore configuration
├── README.md              # Project documentation (this file)
├── python-tutorials/      # Programming tutorials & foundation tasks
│   ├── Beehive_data .csv
│   ├── Intro_to_Pandas.ipynb
│   ├── Intro_to_Python.ipynb
│   ├── Moons_and_Planets.csv
│   └── Numpy.ipynb
├── notebooks/             # Principal research and analysis notebooks
│   └── StrangerPulses.ipynb
└── reports/               # Full scientific reports and project guides
    └── stranger pulses.pdf
```

---

## Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amanpal27/Stranger-Pulses.git
   cd Stranger-Pulses
   ```

2. **Set up virtual environment & install requirements**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```
   *Note: Ensure libraries `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`, `ucimlrepo`, and `mpl_toolkits` are installed.*

3. **Running the Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/StrangerPulses.ipynb
   ```
