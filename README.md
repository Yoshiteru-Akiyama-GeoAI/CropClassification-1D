[🇺🇸 English](README.md) | [🇯🇵 日本語](README_jp.md)

# 🌾 CropClassification-1D Benchmark Study

## Pixel-based Deep Learning Benchmark for Crop Classification Using Sentinel-2 Time Series Data

This repository provides a benchmarking framework for comparing crop classification models using the **TimeSen2Crop** dataset published by Weikmann et al. TimeSen2Crop is a publicly available benchmark dataset designed for crop classification research based on Sentinel-2 satellite time-series data.

The objective of this study is to investigate how different sequence modeling architectures, temporal encoding methods, and temporal aggregation strategies influence crop classification performance.

Three representative deep learning architectures are compared:

* **BiLSTM** (Recurrent Neural Network)
* **Temporal Convolutional Network (TCN)**
* **Transformer Encoder**

Rather than proposing a new model, this repository provides a systematic comparison to better understand how temporal information should be represented and aggregated for satellite-based crop classification.

Unlike previous studies that focus on a single architecture, this repository provides a unified benchmark to systematically evaluate temporal encoding and aggregation strategies across recurrent, convolutional, and attention-based sequence models under identical experimental conditions.

---

# 🎯 Research Objectives

This study addresses the following research questions.

### 1. Which deep learning architecture is most suitable for Sentinel-2 crop time series?

Three representative temporal models are compared.

* BiLSTM
* TCN
* Transformer Encoder

---

### 2. How should acquisition time be represented?

Three temporal encoding strategies are investigated.

| Encoding    | Description                             |
| ----------- | --------------------------------------- |
| **None**    | No explicit temporal information        |
| **DOY**     | Day of Year (absolute acquisition date) |
| **Sin/Cos** | Cyclic sinusoidal representation        |

---

### 3. How should variable-length sequences be aggregated?

Three aggregation strategies are evaluated.

| Aggregation | Description                   |
| ----------- | ----------------------------- |
| Last        | Last valid observation        |
| Attention   | Attention-based pooling       |
| Masked GAP  | Masked Global Average Pooling |

---

# 📌 Overall Pipeline

In this study, crop classification is performed on Sentinel-2 time-series data by systematically applying three components: temporal encoding, temporal sequence models, and temporal aggregation strategies.

<p align="center">
<img src="figures/pipeline.png" width="800">
</p>

---

# 🛰 Dataset

This project uses the publicly available **TimeSen2Crop** dataset for crop type classification.

Each sample consists of

* Sentinel-2 multi-temporal spectral observations
* Crop type label
* Variable-length observation sequence

The input tensor has the following shape:

```text
(T, C)
```

where

* **T** : Number of valid satellite observations
* **C** : Number of input features

Unlike many benchmark datasets, the number of observations varies among samples because Sentinel-2 acquisitions depend on cloud coverage, revisit frequency, and image availability.

Example:

```text
Sample A : (31, C)

Sample B : (36, C)

Sample C : (38, C)
```

Therefore, all models in this repository are designed to process **variable-length temporal sequences** without discarding valid observations.

---

# 📂 Repository Structure

```text
.
├── core/
│   ├── fileIF.py
│   ├── inference.py
│   ├── train.py
│   └── utils.py
│
├── dataset/
│   ├── cache_builder.py
│   └── cache_dataset.py
│
├── models/
│   ├── common.py
│   ├── lstm.py
│   ├── tcn.py
│   └── transformer.py
│
├── pipeline/
│   ├── cache_pipe.py
│   ├── inference_pipe.py
│   └── train_pipe.py
│
├── figures/
│   ├── model_bilstm.png
│   ├── model_tcn.png
│   ├── model_transformer.png
│   ├── pipeline.png
│   ├── temporal_aggregation.png
│   └── temporal_encoding.png
│
├── main.py
├── main.yaml
├── LICENSE
├── requirements.txt
├── README.md
└── README_jp.md
```

> The `figures/` directory contains conceptual illustrations and experimental results used in this README.

---

# ⚙ Installation

## Requirements

* Python ≥ 3.10
* PyTorch
* NumPy
* Pandas
* scikit-learn
* matplotlib
* rasterio
* PyYAML
* tqdm

Install the required packages using:

```bash
pip install -r requirements.txt
```

For GPU support, install the CUDA-compatible PyTorch build from the PyTorch official installation guide.

### Tested with

- Python 3.10
- PyTorch 2.5
- CUDA 12.1

---

# 🚀 Quick Start

The execution mode is controlled by the `proc_type` parameter in `main.yaml`.

| `proc_type` | Description         |
| ----------- | ------------------- |
| 0           | Build cache dataset |
| 1           | Train the model     |
| 2           | Run inference       |

The execution command is the same for all modes.

```bash
python main.py
```

or

```bash
python main.py --yaml_file main.yaml
```

If `--yaml_file` is omitted, `main.yaml` is used by default.

## 1. Build Cache Dataset

Set the following in `main.yaml`:

```yaml
proc_type: 0
```

Then run:

```bash
python main.py
```

---

## 2. Train the Model

Set the following in `main.yaml`:

```yaml
proc_type: 1
```

Then run:

```bash
python main.py
```

---

## 3. Run Inference

Set the following in `main.yaml`:

```yaml
proc_type: 2
```

Then run:

```bash
python main.py
```

---

# 🧠 Model Architecture

This study evaluates three deep learning architectures with different capabilities for modeling temporal dependencies in Sentinel-2 time-series data.

---

## BiLSTM

<p align="center">
<img src="figures/model_bilstm.png" width="600">
</p>

Bidirectional LSTM is used to extract temporal features by exploiting information from both past and future observations.

---

## TCN

<p align="center">
<img src="figures/model_tcn.png" width="600">
</p>

Temporal Convolutional Network (TCN) extracts local temporal patterns through convolution-based sequence modeling.

---

## Transformer Encoder

<p align="center">
<img src="figures/model_transformer.png" width="600">
</p>

Transformer Encoder learns long-range dependencies between observations using self-attention mechanisms.

---

# 📖 Experimental Design

The experimental methodology is organized into three phases.

1. **Phase 1** – Comparison of temporal aggregation strategies.
2. **Phase 2** – Comparison of temporal encoding methods using the best aggregation selected in Phase 1.
3. **Phase 3** – Final comparison of the best-performing configuration for each model architecture.

The benchmark consists of three experimental phases designed to isolate the influence of temporal aggregation, temporal encoding, and model architecture.

---

# Phase 1 — Temporal Aggregation Comparison

## Objective

DOY (Day of Year) encoding was fixed, and the effect of temporal aggregation strategies was evaluated.

Because Sentinel-2 observations form variable-length time series,
the strategy for converting temporal information into classification features is critical.

<p align="center">
<img src="figures/temporal_aggregation.png" width="750">
</p>

---

## Compared Aggregation Methods

| Aggregation | Description                                          |
| ----------- | ---------------------------------------------------- |
| Last        | Use the final valid observation                      |
| Attention   | Learn weighted importance over all observations      |
| Masked GAP  | Average all valid observations using a temporal mask |

Each aggregation method was evaluated using the same training configuration for all three architectures.

---

## Results

| Model           |       Last |  Attention | Masked GAP | Selected   |
| --------------- | ---------: | ---------: | ---------: | ---------- |
| **BiLSTM**      |     0.7445 | **0.8040** |     0.7681 | Attention  |
| **TCN**         |     0.5525 |     0.7753 | **0.8091** | Masked GAP |
| **Transformer** | **0.8410** |     0.8332 |     0.8228 | Last       |

---

## Discussion

Different architectures preferred different aggregation strategies.

* **BiLSTM** benefited from attention pooling, indicating that multiple observations contribute to the final representation.
* **TCN** achieved the highest accuracy using Masked Global Average Pooling, suggesting that convolutional features are best summarized over the complete temporal sequence.
* **Transformer** performed best by simply using the final token representation.

---

# Phase 2 — Temporal Encoding Comparison

## Objective

The optimal aggregation strategy identified in Phase 1 was fixed for each model,
and the impact of temporal encoding strategies was evaluated.

In Sentinel-2 time-series observations, the acquisition date itself provides
important information related to crop phenology.

This study compares the following three temporal representations:

<p align="center">
<img src="figures/temporal_encoding.png" width="750">
</p>

---

The following aggregation strategies were fixed.

| Model       | Fixed Aggregation |
| ----------- | ----------------- |
| BiLSTM      | Attention         |
| TCN         | Masked GAP        |
| Transformer | Last              |

Three temporal encoding strategies were compared.

| Encoding | Description                      |
| -------- | -------------------------------- |
| None     | No explicit temporal information |
| DOY      | Day of Year                      |
| Sin/Cos  | Cyclic sinusoidal representation |

The DOY results obtained in Phase 1 were reused as the baseline for this comparison.

---

## Results

| Model           |       None |        DOY | Sin/Cos | Selected |
| --------------- | ---------: | ---------: | ------: | -------- |
| **BiLSTM**      |     0.7360 | **0.8040** |  0.7725 | DOY      |
| **TCN**         |     0.7996 | **0.8091** |  0.8008 | DOY      |
| **Transformer** | **0.8640** |     0.8410 |  0.7779 | None     |

---

## Discussion

The effectiveness of temporal encoding strongly depended on the underlying architecture.

### BiLSTM

Providing the acquisition date explicitly improved classification accuracy substantially.

The recurrent architecture appears to benefit from absolute seasonal information that helps distinguish crop growth stages.

---

### TCN

Temporal encoding produced only a small improvement.

The temporal convolution itself already captures most phenological patterns contained in the observation sequence.

---

### Transformer

The best performance was obtained **without explicit temporal encoding**.

Adding DOY slightly reduced accuracy, while sinusoidal encoding caused a much larger degradation.

This suggests that self-attention effectively models temporal relationships directly from the observation sequence.

---

# Phase 3 — Final Model Comparison

The best configuration for each architecture was selected based on Phase 1 and Phase 2.

| Rank | Model       | Encoding | Aggregation | Overall Accuracy |
| ---: | ----------- | -------- | ----------- | ---------------: |
|   🥇 | Transformer | None     | Last        |       **0.8640** |
|   🥈 | TCN         | DOY      | Masked GAP  |       **0.8091** |
|   🥉 | BiLSTM      | DOY      | Attention   |       **0.8040** |

---

# 📊 Experimental Summary

The benchmark required a total of **15 training runs**.

| Phase   | Objective                    |                      Runs |
| ------- | ---------------------------- | ------------------------: |
| Phase 1 | Aggregation comparison       |                         9 |
| Phase 2 | Temporal encoding comparison |  6 *(DOY results reused)* |
| Phase 3 | Final comparison             | Reuse best configurations |

This staged design isolates the contribution of each component while avoiding redundant experiments.

---

# 🔍 Key Findings

## 1. Optimal temporal aggregation depends on the model architecture.

No aggregation strategy consistently outperformed the others.

* BiLSTM → Attention
* TCN → Masked GAP
* Transformer → Last

This indicates that sequence aggregation should be selected according to the characteristics of each backbone model.

---

## 2. Explicit temporal information is not universally beneficial.

DOY encoding substantially improved BiLSTM performance and slightly improved TCN performance.

However, the Transformer achieved its highest accuracy without any explicit temporal encoding.

---

## 3. Sin/Cos encoding was less effective for crop phenology.

Sinusoidal encoding assumes cyclic temporal behavior.

In contrast, crop development within a single growing season follows a monotonic phenological progression.

Although Sin/Cos improved several individual crop classes, it consistently underperformed DOY in overall accuracy.

---

## 4. Transformer achieved the highest overall performance.

Among all evaluated configurations,

**Transformer + No Temporal Encoding + Last Aggregation**

obtained the highest validation accuracy of **86.40%**.

This result suggests that self-attention can effectively capture temporal dependencies directly from Sentinel-2 observation sequences.

---

# 🌱 Conclusions

This benchmark systematically compared temporal modeling strategies for Sentinel-2 crop classification.

The experiments demonstrate that:

* The optimal temporal representation depends on the model architecture.
* Explicit temporal information is beneficial for recurrent models but unnecessary for the evaluated Transformer.
* Aggregation strategy plays a critical role for variable-length satellite time series.
* Self-attention provides the strongest overall performance among the evaluated architectures.

These findings provide practical guidance for selecting temporal modeling strategies in satellite-based crop classification.

---

# 🌱 Future Work

The following directions will be explored as future extensions of this study.

## 1. Investigation of Advanced Temporal Encoding Methods

We will investigate temporal representations that combine monotonic seasonal progression, such as Day of Year (DOY), with periodic representations such as Sin/Cos encoding.

Examples include:

* Learnable Temporal Embedding
* DOY + Seasonal Component
* Phenological Phase Encoding

---

## 2. Adaptation to Regional and Cropping-Season Variations

The optimal temporal representation may vary across different regions and agricultural fields with multiple cropping seasons.

Future work will evaluate the proposed approaches using larger and more diverse datasets covering broader geographical areas.

---

## 3. Comparison with 2D Spatio-Temporal Models

This study focuses on pixel-based time series classification using Sentinel-2 temporal data. Future work will investigate 2D deep learning models that incorporate spatial information, including:

* U-Net
* BiLSTM-U-Net
* Vision Transformer

These approaches will be applied to crop and land cover classification and compared with the current pixel-based framework to systematically evaluate the differences in how temporal and spatial information are utilized.

---

# 📚 Citation

If you use this repository or the experimental protocol, please consider citing this project together with the original **TimeSen2Crop** dataset publication.

```bibtex
@misc{cropclassification_1d_benchmark,
  title={CropClassification-1D Benchmark: Pixel-based Deep Learning Benchmark for Crop Classification Using Sentinel-2 Time Series Data},
  author={Yoshiteru Akiyama},
  year={2026},
  publisher={GitHub},
  howpublished={https://github.com/Yoshiteru-Akiyama-GeoAI/CropClassification-1D.git},
  note={GitHub Repository}
}
```

## Dataset

This project uses the TimeSen2Crop dataset.
Please also cite the original TimeSen2Crop dataset paper:

```bibtex
@article{weikmann2021timesen2crop,
  title={TimeSen2Crop: A Million Labeled Samples Dataset of Sentinel 2 Image Time Series for Crop-Type Classification},
  author={Weikmann, Giulio and Paris, Claudia and Bruzzone, Lorenzo},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={14},
  pages={4699--4708},
  year={2021},
  doi={10.1109/JSTARS.2021.3073965}
}
```

---

# 📄 License

This project is released under the **MIT License**.

Copyright (c) 2026 Yoshiteru Akiyama

The TimeSen2Crop dataset is not included in this repository and remains subject to the original dataset license and publication terms.

See the `LICENSE` file for details.
