# GitHub Repository for "Machine Learning-Guided Discovery of High-Entropy Perovskite Oxide Electrocatalysts via Oxygen Vacancy Engineering"

This repository contains the complete source code and documentation for the OxiGraphX model described in our published paper:

> Tukur et al., "Machine Learning‐Guided Discovery of High‐Entropy Perovskite Oxide Electrocatalysts via Oxygen Vacancy Engineering," *Small* (2025).

## Repository Overview

```
repository/
├── main.py             # Main script for training, evaluation, data loading, and model initialization
├── train.py            # Training and evaluation routines
├── dataset.py          # Data handling and processing with PyTorch Geometric
├── ceal.py             # Implementation of the CEAL convolutional layer
├── model.py            # OxiGraphX (MyCEALNetwork) graph neural network model
├── pred.py             # Prediction script for new datasets
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
