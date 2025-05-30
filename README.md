# GitHub Repository for "Machine Learning-Guided Discovery of High-Entropy Perovskite Oxide Electrocatalysts via Oxygen Vacancy Engineering"

## 🔬 Introduction

This repository accompanies the paper:

**[Tukur et al., "Machine Learning‐Guided Discovery of High‐Entropy Perovskite Oxide Electrocatalysts via Oxygen Vacancy Engineering," _Small_ (2025)](https://onlinelibrary.wiley.com/doi/10.1002/smll.202501946)**.  

It contains the source code and data pipeline for building and training the **OxiGraphX** model — a novel graph neural network framework based on the CEAL-WF layer for accurate prediction of oxygen vacancy formation energies (OVFEs) in high-entropy perovskite oxides.

**OxiGraphX**, is introduced as a novel graph neural network (GNN) model designed to capture the complex relationships among structure, composition, and atomic chemical environments for accurate prediction of oxygen vacancy formation energies (OVFEs) in HEPOs. By integrating machine learning (ML), density functional theory (DFT), and experimental validation, this model demonstrates an efficient framework for rapidly and accurately screening HEPO electrocatalysts for oxygen evolution reaction (OER).

Figure 1. below depicts the material development framework encompassing DFT calculations, ML predictions, and experimental validations. To the best of our knowledge, this framework presents a pioneering effort that translates ML outcomes into laboratory experiments. This innovative framework empowers researchers to leverage ML models to predict diverse properties essential for experimental studies.
<img src="https://github.com/user-attachments/assets/ad27885c-7d4c-46ed-8c4d-ca0ef06c3777" alt="Sample Image" width="500">

Figure1. Integrated framework featuring ML, DFT, and experiments for OVFE prediction in HEPOs.

OxiGraphX uses a Chemical Environment Adaptive Learning layer with Learnable Weighting Functions (CEAL-WF), shown in Figure 2. It integrates various aggregation functions (e.g., mean, sum, max) and learnable scalers to capture the complex atomic interactions and compositional diversity in HEPOs.

![image](https://github.com/user-attachments/assets/bbff78df-190d-4c42-8878-3014c4e617ea|width=957)

Figure 2. a) The architecture of the CEAL-WF convolutional layer. b) Overview of OxiGraphX model for OVFE prediction in HEPOs.

## Repository Overview

```
repository/
├── main.py             # Main script for training, evaluation, data loading, and model initialization
├── train.py            # Training and evaluation routines
├── dataset.py          # Data handling and processing with PyTorch Geometric
├── config.py          # dataset configurations
├── ceal.py             # Implementation of the CEAL convolutional layer
├── model.py            # OxiGraphX (MyCEALNetwork) graph neural network model
├── pred.py             # Use a trained model to predict for new datasets
├── scalers.py          # Degree-based scaler functions
├── aggregators.py      # aggregator functions
├── utils.py            # General utility functions (training, evaluation, visualization)
├── utils_data_JSNN.py  # Project-specific data preprocessing utilities
└── data/               # data foler
```

## Data Preparation

Unzip the **data.tar.gz** file in the directory. The dataset shall be prepared following the structure below:

```
data/
  ├── Compound_Name/
  │     ├── CONFIGS/
  │          ├── CRYSTAL_*.xyz
  │     └── DEFECT_ENERGY_EV
  └── Another_Compound/
  │     ├── CONFIGS/
  │          ├── CRYSTAL_*.xyz
  │     └── DEFECT_ENERGY_EV
```

Each compound directory must contain atomic coordinates (CRYSTAL_*.xyz) and energy values (`DEFECT_ENERGY_EV`).

## Requirements

The **env.yaml** describes the environment requirements.

You can also install the necessary Python libraries using pip:

```bash
pip install torch torch-geometric numpy matplotlib sklearn scipy tqdm
```


## Training the Model

Before training, set the following environment variables either directly in your terminal or via a bash script. An example is provided in job.pbs

```bash
export numDatasets=6
export epochs=2500
export learning_rate=0.001
export batch_size=32
export train_dataset_idx='[0,1,2,3,4,5]'
export test_dataset_idx='[0,1,2,3,4,5]'
export dispProgress=True
export numLayers=5
```

Then start the training process:

```bash
python main.py
```

## Performing Predictions

To use the trained model for predictions:

1. Update the path to your trained model in `pred.py`:

```python
path = '/path/to/your/trained/model.pth'
```

2. Execute the prediction script:

```bash
python pred.py
```

## Citation

If you use this code for your research, please cite our paper:

```
@article{tukur2025machine,
  author={Tukur, Panesun and Wei, Yong and Zhang, Yinning and Chen, Hanning and Lin, Yuewei and He, Selena and Mo, Yirong and Wei, Jianjun},
  title={Machine Learning‐Guided Discovery of High‐Entropy Perovskite Oxide Electrocatalysts via Oxygen Vacancy Engineering},
  journal={Small},
  year={2025},
  doi={10.1002/smll.202501946}
}
```

## Contact

For any inquiries or assistance with the code, please open a GitHub issue or contact:

* Yinning Zhang: [yzhang@westga.edu](mailto:yzhang@westga.edu)
* Dr. Yong Wei: [yong.wei@ung.edu](mailto:yong.wei@ung.edu)

We welcome your feedback and collaboration. Thank you!
