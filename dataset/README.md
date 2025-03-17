# Dataset Documentation

## TBI (Traumatic Brain Injury) Dataset

This folder contains the TBI (Traumatic Brain Injury) dataset used for classification tasks.

### Dataset Description

The dataset (`tbi.csv`) contains clinical information about TBI patients with the following key features:

#### Patient Demographics
- `age_at_record`: Patient's age
- `sex`: Patient's gender (0: Male, 1: Female)

#### Clinical Features
- `tbi_cli_reason`: Reason for TBI
- `tbi_cli_time_acci_hos`: Time from accident to hospital (hours)
- `tbi_cli_pulse`: Pulse rate
- `tbi_cli_temp`: Body temperature
- `tbi_cli_blood_pressure`: Blood pressure (systolic/diastolic)
- `tbi_cli_breathing_rate`: Breathing rate
- `tbi_cli_glasgow`: Glasgow Coma Scale score (3-15)

#### Clinical Symptoms
- Various binary indicators (1: Yes, 2: No) for:
  - `tbi_cli_awaken`: Consciousness
  - `tbi_cli_headache`: Headache
  - `tbi_cli_blue`: Cyanosis
  - `tbi_cli_para_ner`: Paralysis
  - And other neurological symptoms

#### Laboratory Tests
- `hong_cau_v2`: Red blood cell count
- `bach_cau_v2`: White blood cell count
- `tieu_cau_v2`: Platelet count
- And other blood test results

#### Target Variable
- `d_kl_tl`: TBI severity classification (1-4)
  - 1: Mild
  - 2: Moderate
  - 3: Severe
  - 4: Very Severe

### Data Format
- File format: CSV
- Number of instances: ~500
- Number of features: 70+
- Missing values: Yes (marked as empty cells)

### Usage
This dataset is used for:
1. TBI severity classification
2. Clinical outcome prediction
3. Risk factor analysis

### Data Preprocessing
See `dataset_confinement.ipynb` for data preprocessing steps including:
- Handling missing values
- Feature scaling
- Feature selection
- Data validation

### Citation
If you use this dataset, please cite: [Add citation information] 