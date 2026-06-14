# Pulsemind BKI Model Training

This repository contains the training scripts and methodologies for developing machine learning models to predict in-hospital mortality (`in_hospital_death`) based on patient ventilator and waveform data.

## 1. Initial Training & Discovery

Initially, we developed two models to process the patient waveform data (Flow, Pao, and Pes):
* A **Scikit-Learn Gradient Boosting Classifier** utilizing CPU-parallelization.
* A **PyTorch Multi-Layer Perceptron (MLP)** utilizing the NVIDIA H200 GPU.

**Initial Results:** Both models quickly achieved > 99% accuracy.
**The Catch:** This exceptionally high accuracy was a false positive caused by **Data Leakage**. The data was randomly split into 80% training and 20% testing sets using a standard `train_test_split`. Because the data represents continuous time-series waveforms from only 7 patients, shuffling the rows meant that the model saw 80% of a patient's timeline in training, and was evaluated on the remaining 20% of that same patient's timeline. It learned to *memorize* the specific patients rather than generalizing clinical signs of mortality.

## 2. Superior Methodology (Advanced Pipeline)

To output better, scientifically valid data and fix the leakage, we implemented a superior methodology in `train_bki_models_advanced.py`:

### A. Patient-Level Splitting (Leave-One-Group-Out)
We discarded randomized splitting in favor of strict patient-level grouping. We separated the data so that training, validation, and testing sets never share the same patient. This ensures the model is tested on a truly "unseen" patient, correctly measuring its real-world generalization capability.

### B. Time-Series Sequence Modeling
Raw waveforms are sequential, not static. We introduced a sliding-window data generator that processes the data in sequential blocks (e.g., 100 timesteps at a time) rather than as isolated rows.

### C. 1D-Convolutional Neural Network (1D-CNN)
We upgraded the architecture from a simple MLP to a PyTorch 1D-CNN. This allows the model to extract features from the *shape*, *frequency*, and *patterns* of the pressure and flow waves over time.

### D. Early Stopping & Optimization
To perfectly utilize the H200 GPU's capability without overfitting, we implemented **Early Stopping**. Instead of guessing a "golden ratio" of samples to epochs, the model trains indefinitely but monitors a validation dataset. If the validation loss fails to improve for a set number of epochs (patience), training automatically halts, saving the best iteration. 

We also implemented class-weighting in the `BCEWithLogitsLoss` function to penalize the model more heavily for missing positive mortality cases, compensating for the imbalanced dataset (only 2 out of 7 patients died).

## 3. Results & Next Steps

When evaluating the superior 1D-CNN model on a completely isolated, unseen test patient (P02), the accuracy fell dramatically compared to the initial flawed models. 

**Key Finding:** The model completely failed to generalize the signs of mortality to an unseen patient. This is an expected and vital clinical finding. It proves that a dataset of only 7 patients is far too small for deep learning models to identify generalized physiological markers of mortality. 

**Next Steps:**
To achieve a viable clinical model, the primary requirement is a massive scale-up in the dataset. We must acquire waveform data from hundreds or thousands of patients. The advanced 1D-CNN architecture and patient-level splitting pipeline established here are fully prepared to train on that scaled dataset without introducing data leakage.

## Usage

To run the advanced methodology script on the cluster:
```bash
source .venv/bin/activate
python3 train_bki_models_advanced.py
```
