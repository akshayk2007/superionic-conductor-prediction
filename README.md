# LiScreen – Superionic Conductor Prediction

**Machine Learning-Based Screening of Lithium-Ion Solid Electrolyte Materials**

## 1. Overview

LiScreen is a machine learning-based web application designed to screen potential lithium-ion solid electrolyte materials and classify them as superionic conductors or poor conductors.

The application uses crystallographic information from CIF (Crystallographic Information File) structures and five material descriptors to make predictions using a trained Random Forest classification model.

## 2. Problem Statement

Identifying promising solid electrolyte materials for next-generation batteries requires extensive material analysis and experimental evaluation.

LiScreen aims to simplify the initial screening process by using crystallographic descriptors and machine learning to identify potentially promising lithium-ion conductors.

## 3. Key Features

- **CIF File Upload:** Upload crystal structure files for automated descriptor extraction and prediction.
- **CSV Batch Screening:** Upload CSV files containing multiple materials and their descriptors.
- **Manual Prediction:** Enter the five descriptors manually.
- **Random Forest Classification:** Predict whether a material belongs to the superionic or poor conductor category.
- **Prediction Probability:** Display the model's estimated probability for the superionic class.
- **Structure Information:** Display extracted crystal structure information and calculated descriptors.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn |
| ML Algorithm | Random Forest Classifier |
| CIF Processing | Pymatgen |
| Data Processing | NumPy, Pandas |
| Model Serialization | Joblib |

## 5. Five Crystallographic Descriptors

The model uses five descriptors:

- **LASD:** Lithium–Anion Shortest Distance
- **SBI:** Structure Bonding Indicator
- **AFC:** Anion Framework Connectivity
- **LLSD:** Lithium–Lithium Shortest Distance
- **LLB:** Lithium–Lithium Bonding

These descriptors are used as input features for the trained machine learning model.

## 6. System Workflow

1. User uploads a CIF file, CSV file, or enters descriptors manually.
2. For CIF input, Pymatgen parses the crystal structure.
3. The application extracts the five crystallographic descriptors.
4. The descriptors are passed to the trained Random Forest model.
5. The model generates a classification prediction and superionic-class probability.
6. The frontend displays the prediction results.

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- pip

### Clone the repository

```bash
git clone https://github.com/akshayk2007/superionic-conductor-prediction.git
cd superionic-conductor-prediction
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
uvicorn main:app --reload
```

Open your browser and visit:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## 8. Project Structure

```text
superionic-conductor-prediction/
│
├── main.py
├── index.html
├── style.css
├── script.js
├── logo.png
├── rf_model.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend |
| `/predict` | POST | Manual descriptor prediction |
| `/predict-cif` | POST | CIF file prediction |
| `/predict-csv` | POST | Batch CSV prediction |
| `/health` | GET | Checks backend and model status |

## 10. Limitations

- Predictions depend on the quality and distribution of the training dataset.
- CIF descriptor extraction uses geometric and occupancy-based calculations.
- Disordered crystal structures use occupancy-weighted approximations.
- The extracted descriptors may differ from the exact procedure used to generate the training features.
- Model predictions are screening results and require experimental or independent scientific validation.

## 11. Future Improvements

- Improve descriptor extraction and validate it against reference datasets.
- Expand the training dataset with additional materials.
- Improve model performance through systematic evaluation.
- Add more crystallographic and material properties.
- Improve prediction explainability.

## 12. Contributors

Add your project team members and their respective contributions here.

## 13. Disclaimer

LiScreen is intended as a research and educational screening tool. Its predictions should not be treated as experimentally confirmed conductivity measurements.
