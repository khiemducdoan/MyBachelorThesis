# MyBachelorThesis AI Pipeline

This project provides a comprehensive AI pipeline for processing and analyzing datasets, specifically designed for Traumatic Brain Injury (TBI) data. The goal is to leverage machine learning models to predict and analyze TBI outcomes. The pipeline is flexible and configurable, allowing users to adapt it to their own datasets and models.

---

## Project Structure

- **configs/**: Configuration files
  - **model/**: Model configurations
  - **data/**: Dataset configurations
  - **experiment/**: Experiment configurations
  - **default.yaml**: Default settings

- **dataset/**: Dataset files and preprocessing
  - **raw/**: Raw data files
  - **processed/**: Processed data files
  - **preprocessing/**: Preprocessing scripts

- **src/**: Source code
  - **models/**: Model implementations
  - **data/**: Data handling
  - **utils/**: Utility functions
  - **train.py**: Training script

- **outputs/**: Model outputs and logs

- **README.md**: Project documentation

---

## Installation Guide

### Prerequisites

Ensure you have Python 3.8+ installed. You will also need to install the necessary Python packages.

### Setting Up the Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Pipeline

### 1. Prepare Your Data

- **Place your raw data files** in the `dataset/raw/` directory.
- **Run the preprocessing script** to clean and prepare your data:
  ```bash
  python dataset/preprocessing/prepare_data.py
  ```

### 2. Configure Your Experiment

- **Edit configuration files** in `configs/` to set up your experiment. You can specify:
  - The model to use (e.g., TabNet, NAIM)
  - Training parameters (e.g., batch size, learning rate)
  - Dataset details

### 3. Train the Model

- **Run the training script** with your configuration:
  ```bash
  python src/train.py experiment=your_experiment
  ```

### 4. Evaluate and Analyze

- **After training**, evaluate the model's performance using the evaluation scripts or notebooks provided in the `src/` directory.

### 5. View Results

- **Check the `outputs/` directory** for:
  - Model checkpoints
  - Training logs
  - Evaluation metrics

---

## Notes & Troubleshooting

- **Common Issues**:
  - Ensure all paths in configuration files are correct.
  - Check for missing values and data format inconsistencies.

- **Performance Tuning**:
  - Experiment with different models and hyperparameters.
  - Validate data preprocessing steps.

For further assistance, please open an issue on the project's GitHub repository.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
