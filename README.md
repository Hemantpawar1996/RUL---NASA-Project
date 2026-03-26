## 📄 README.md

```markdown
# 🚀 RUL Prediction using NASA Dataset

## 📌 Overview
This project predicts the **Remaining Useful Life (RUL)** of aircraft engines using the NASA Turbofan Engine Degradation dataset.  
It applies deep learning models such as **LSTM, GRU, and CNN** for predictive maintenance.

The goal is to estimate how many cycles an engine can run before failure.

---

## ⚙️ Features
- Data preprocessing and scaling
- RUL calculation for training and test datasets
- Exploratory Data Analysis (EDA)
- Sequence generation for time-series modeling
- Deep learning models:
  - LSTM
  - GRU
  - CNN
- Performance evaluation using:
  - RMSE
  - MAE
  - R² Score
  - NASA Score
- Visualization of:
  - Loss curves
  - Predictions vs actual values
  - Error distribution

---

## 📂 Project Structure
```

├── main.py
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
├── results/
│   ├── plots and metrics

````

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/RUL-NASA-Project.git
cd RUL-NASA-Project
````

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib scikit-learn torch
```

---

## 🧪 Required Libraries

Your project uses the following Python libraries:

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `torch` (PyTorch)
* `argparse`
* `pathlib`

---

## ▶️ How to Run

```bash
python main.py
```

### Optional arguments:

```bash
python main.py --epochs 20 --batch 512 --window 30 --subset FD001
```

---

## 📊 Dataset

NASA Turbofan Engine Degradation Dataset:

* Train: `train_FD001.txt`
* Test: `test_FD001.txt`
* RUL labels: `RUL_FD001.txt`

---

## 🧠 Models Used

### 1. LSTM

Captures long-term dependencies in time-series data.

### 2. GRU

Efficient alternative to LSTM with fewer parameters.

### 3. CNN

Extracts temporal patterns using convolution layers.

---

## 📈 Evaluation Metrics

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **R² Score**
* **NASA Score** (custom metric for RUL tasks)

---

## 📊 Output

Results are saved in the `results/` folder:

* Training loss plots
* Prediction scatter plots
* Error distribution
* Model comparison CSV

---

## 🎯 Objective

To improve predictive maintenance by accurately estimating engine failure time using machine learning models.

---
