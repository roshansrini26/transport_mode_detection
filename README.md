# Transport Mode Detection from Sensor Data

This project focuses on detecting the user's mode of transport (Car, Still) using smartphone sensor data.  
It involves two key stages:
- Pre-processing and feature extraction from raw sensor data.
- Training Machine Learning models for transport mode classification.

The project includes Python scripts for **data preprocessing** and a **Jupyter Notebook** for **model training and evaluation**.

---

## Dependencies

Make sure you have the following installed:
- Python 3.7+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib (for evaluation plots)

Install required libraries using:

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

## Documentation
### Code
In this section we show the functionalities developed in our work and the relative parameters used.
#### TMDataset.py
<table>
<thead>
<th>Module name</th>
<th>Parameter</th>
<th>Description</th>
</thead>
<tbody>
<tr>
<td>load_data(data
-dir)</td>
<td>data_dir (str): path to directory containing raw .csv files</td>
<td>
Loads all ride data into a single pandas DataFrame.
</td>
</tr>

<tr>
<td>clean_data(df)</td>
<td>df (DataFrame): raw ride data</td>
<td>Removes missing values and outliers using Z-score method.</td>
</tr>

<tr>
<td>compute_orientation(df)</td>
<td>df (DataFrame): cleaned ride data</td>
<td>Calculates roll, pitch, game rotation vector, and orientation using a simplified complementary filter.</td>
</tr>

<tr>
<td>calculate_statistics(df)</td>
<td>df (DataFrame): ride data with orientation</td>
<td>Generates statistical features per ride (mean, min, max, std) for each sensor.</td>
</tr>

<tr>
<td>main()</td>
<td></td>
<td>Main execution flow: load data, clean, engineer features, and save to a standardized CSV.</td>
</tr>

</tbody>
</table>

### Output

The preprocessing script generates:

- **standardized_ride_data.csv**:  
  A clean and feature-engineered dataset ready for model training.

### Model Training Code (TransportModeDetection.ipynb)

The Jupyter Notebook covers the complete machine learning pipeline:

- **Loading the standardized dataset** (`standardized_ride_data.csv`)
- **Splitting the dataset** into training and testing sets
- **Model Training** using:
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- **Model Evaluation** using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)

The notebook demonstrates **end-to-end training and testing** for transport mode classification based on smartphone sensor data.

## How to Run

### Step 1: Preprocess Raw Data

Place your ride data `.csv` files in a folder (e.g., `ride_data/`).

Then execute the preprocessing script:

```bash
python preprocessing.py
```

This will generate the `standardized_ride_data.csv` file.

---

### Step 2: Train and Evaluate Models

Open the Jupyter Notebook:

```bash
jupyter notebook TransportModeDetection.ipynb
```

Follow the steps inside the notebook to:

- **Load the dataset**
- **Train machine learning models**
- **Evaluate model performance**


###Project Structure
```unicode
.
├── ride_data/                     # Folder containing raw ride data (.csv files)
├── standardized_ride_data.csv      # Output dataset after preprocessing
├── preprocessing.py                # Python script for data preprocessing
├── TransportModeDetection.ipynb    # Jupyter Notebook for model training and evaluation
└── README.md                       # Project documentation
```
## License
This work is licensed under a MIT License.

## Team of collaborators
Lakshman Navaneetha Krishnan and Roshan