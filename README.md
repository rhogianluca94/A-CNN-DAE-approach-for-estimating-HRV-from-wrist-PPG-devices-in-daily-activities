# A Deep Learning approach for estimating Time-Domain Heart Rate Variability parameters from wrist photoplethysmography during daily activity

This repository contains the code associated with the paper:

> Rho, G., Carbonaro, N., Laurino, M., Tognetti, A., & Greco, A. (2026). A deep learning approach for estimating time-domain heart rate variability parameters from wrist photoplethysmography during daily activity. *Biomedical Signal Processing and Control, 123*, 110531. https://doi.org/10.1016/j.bspc.2026.110531

The code denoises wrist photoplethysmography (PPG) signals corrupted by motion artifacts using an ensemble of convolutional denoising autoencoders (CNN-DAEs), and then estimates time-domain pulse rate variability (PRV) parameters (mean heart rate, stdRR, RMSSD) with a dedicated tracking algorithm.

## Overview

Wrist-worn devices (smartwatches, fitness bands, research-grade wearables) measure the pulse using PPG. However, movement artifacts severely degrade the signal, which makes it difficult to estimate not only heart rate but also the beat-to-beat variability that carries information about autonomic nervous system activity.

The proposed method consists of two main stages:

1. **Denoising.** An ensemble of 10 CNN-DAEs reconstructs a clean PPG waveform from the noisy input. The networks were trained with a custom loss function that accounts for both the waveform fidelity and the physiological plausibility of the inter-beat intervals derived from the reconstructed signal.
2. **PRV estimation and tracking.** Pulse peaks are detected on the denoised signal (multiscale peak and trough detection, MSPTD), and the resulting inter-beat intervals are used to compute mean heart rate (meanHR), standard deviation of RR intervals (stdRR) and root mean square of successive differences (RMSSD). A tracking algorithm then corrects implausible estimates by combining error detection, robust re-estimation (median), weighted linear regression over neighboring windows, and error propagation control.

The method was validated on two public datasets, IEEESPC and PPG-Dalia. Please refer to the paper for the full methodology and results.

## For Non-Specialists

**What is PPG?** Photoplethysmography is an optical technique that measures blood volume changes under the skin. It is the sensing technology used by most smartwatches to estimate heart rate.

**What is PRV/HRV?** Heart rate variability (HRV) describes the small fluctuations in the time between consecutive heartbeats. It is widely used as an indicator of autonomic nervous system regulation. Pulse rate variability (PRV) is the same concept measured from the pulse wave (PPG) instead of the electrocardiogram (ECG).

**What does this code produce?** For every N-second window of data (default: N=10), it estimates:

- **meanHR**: average heart rate (beats per minute);
- **stdRR**: standard deviation of the intervals between beats (milliseconds^2), reflecting overall variability;
- **RMSSD**: root mean square of successive interval differences (milliseconds^2), commonly associated with parasympathetic activity.

**Does it work with my device?** The algorithms in this repository were designed to operate at 125 Hz. To make the code usable with data from different devices, the preprocessing step **automatically resamples the input signal to 125 Hz** (cubic interpolation) whenever the original sampling rate differs. The only requirement is that you declare the correct original sampling rate of your recording when creating the `PPGdenoising` object:

```python
ppgdenoising = PPGdenoising(srateOrig=64)   # replace 64 with your device's sampling rate in Hz
```

No manual resampling is needed. Note that the pretrained models were trained on specific devices and datasets, so performance on signals with a markedly different pulse morphology may be lower - but we are currently working on this issue (see the generalization analysis in the paper).

## Repository Structure

| File / Folder | Description |
|---|---|
| `main_denoise_ppg_data.py` | Example script: loads a PPG recording, denoises it, extracts PRV parameters and plots the results. |
| `PPGdenoising.py` | Main class `PPGdenoising`: preprocessing, ensemble inference, peak detection and PRV feature extraction. |
| `trackingAlgorithm.py` | Class `PRVtracking`: PRV parameter tracking algorithm (error detection, correction and control). |
| `CNN_IEEESPCData.py` | Definition of the `DAE_CNN` network architecture (PyTorch). |
| `models/` | Pretrained weights of the 10 ensemble models (`model__LOO1_bestModel_M1.h5` ... `_M10.h5`). |
| `proposed/sampleData.csv` | Sample PPG recording (column `ppgData`) used by the example script. |

## Requirements

- Python 3.10 or later (the tracking module uses the `match` statement)
- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- PyTorch

### Quick start

From the repository root, run:

```bash
python main_denoise_ppg_data.py
```

The script loads `proposed/sampleData.csv`, denoises the signal, plots the raw and denoised PPG, and plots the estimated meanHR, stdRR and RMSSD.

### Using the code on your own data

```python
import pandas as pd
from PPGdenoising import PPGdenoising

# 1. Load your PPG signal (1D array of raw samples)
ppg_raw = pd.read_csv('your_file.csv')['ppgData']

# 2. Declare the original sampling rate of your device (Hz)
ppgdenoising = PPGdenoising(srateOrig=64)

# 3. Preprocess: resampling to 125 Hz, band-pass filtering, 8 s windowing, normalization
ppg_prep = ppgdenoising.prepData(data_ppg=ppg_raw, winLen=8, overlap=0)

# 4. Denoise with the CNN-DAE ensemble
ppg_den, ppg_noisy = ppgdenoising.runDenoisingModel(ppg_data=ppg_prep)

# 5. Extract PRV parameters with the tracking algorithm
hr, stdrr, rmssd = ppgdenoising.getPRVfeatures(ppg_data=ppg_den, step=None)
```

### Main parameters

- `srateOrig`: original sampling rate of the input PPG, in Hz.
- `winLen`: window length in seconds. Although 8 s was adopted in the paper, the code uses 10 s for the sake of practicality. 
- `overlap`: overlap between consecutive windows, expressed in percent (0 = no overlap). Use 0 for prediction.
- `step` (in `getPRVfeatures`): length in seconds of the chunks of signal processed at once by the tracking algorithm. If `None`, the whole signal is processed in chunks of at most 30 minutes. Values lower than 30 s are not accepted.

## Processing Pipeline in Detail

1. **Resampling** to 125 Hz (cubic interpolation) if the original sampling rate differs.
2. **Band-pass filtering** with an 8th-order Butterworth filter (0.7-8 Hz).
3. **Segmentation** into 8-second windows (1000 samples at 125 Hz).
4. **Normalization** of each window to the range [0, 1].
5. **Denoising** with the ensemble of 10 CNN-DAEs. The outputs of the models are combined into a single denoised signal.
6. **Peak detection** on the denoised signal with the MSPTD algorithm to obtain inter-beat intervals.
7. **PRV tracking** of meanHR, stdRR and RMSSD, with the threshold parameters `a`, `b` and the maximum number of consecutive errors `theta` defined in `getPRVfeatures`.

## Outputs

- `runDenoisingModel` returns the denoised PPG signal and the corresponding normalized input signal, both concatenated across windows.
- `getPRVfeatures` returns three arrays, one value per analysis window:
  - `hr`: mean heart rate (bpm)
  - `stdrr`: standard deviation of RR intervals (ms)
  - `rmssd`: RMSSD (ms)

## Notes and Limitations

- The pretrained models were trained on wrist PPG data from the IEEESPC and PPG-Dalia datasets. Performance on other devices or populations may differ.
- As reported in the paper, meanHR estimates are reliable during daily-life activities, whereas stdRR and RMSSD estimates are less accurate and should be interpreted with caution.
- PRV parameters estimated from PPG may differ from HRV parameters estimated from ECG, owing to physiological factors such as variations in pulse transit time.
- The method targets time-domain parameters; frequency-domain parameters (LF, HF, LF/HF) are not supported. They can be easily computed starting from the denoised PPG signal, but we do not guarantee on their reliability
- The code is provided for research purposes and is not a medical device.

## Citation

If you use this code or find the associated paper useful, please cite:

**APA**

> Rho, G., Carbonaro, N., Laurino, M., Tognetti, A., & Greco, A. (2026). A deep learning approach for estimating time-domain heart rate variability parameters from wrist photoplethysmography during daily activity. *Biomedical Signal Processing and Control, 123*, 110531. https://doi.org/10.1016/j.bspc.2026.110531

**BibTeX**

```bibtex
@article{rho2026deep,
  title   = {A deep learning approach for estimating time-domain heart rate variability parameters from wrist photoplethysmography during daily activity},
  author  = {Rho, Gianluca and Carbonaro, Nicola and Laurino, Marco and Tognetti, Alessandro and Greco, Alberto},
  journal = {Biomedical Signal Processing and Control},
  volume  = {123},
  pages   = {110531},
  year    = {2026},
  doi     = {10.1016/j.bspc.2026.110531}
}
```

## Data Availability

The code and related material are also archived on Zenodo: https://doi.org/10.5281/zenodo.15595033

The datasets used in the paper are publicly available: the IEEE Signal Processing Cup 2015 (IEEESPC) dataset and the PPG-Dalia dataset.

## License

The material in this repository is published under the GPL-3.0 license.

## Contact

Gianluca Rho - Department of Information Engineering, University of Pisa
Email: gianluca.rho@ing.unipi.it
