# Capstone Module 4 — Construction Safety (Object Detection)

**Project goal:** This project focuses on detecting the completeness of personal protective equipment (PPE) worn by construction workers using YOLO-based object detection.

The pipeline includes dataset preparation, model training, evaluation, inference, and a Streamlit application for real-time detection.


## Key Features
- PPE Detection (person, helmet, vest, no-helmet, no-vest.)
- Training and evaluation notebooks
- Streamlit applications (app.py & app_withAug.py)
- Dataset configuration files (data.yaml & data_augmented.yaml)
## Folder Structure
- extracted/
- models/ (contains best.pt)
- results/
- 01_EDA.ipynb
- 02_Training.ipynb
- 02A_Aug.ipynb
- 02B_Train With Aug.ipynb
- 03_Evaluation.ipynb
- 03_Evaluation_Aug.ipynb
- app.py
- app_withAug.py
- data.yaml
- data_augmented.yaml
- requirements.txt
- training_summary.json
- training_summary_augmented.json
## Training Instructions
Use the following notebooks:
- 02_Training.ipynb (without augmentation)
- 02A_Aug.ipynb + 02B_Train With Aug.ipynb (with data augmentation)
## Evaluation
- 03_Evaluation.ipynb
- 03_Evaluation_Aug.ipynb
## Running Streamlit Applications
Without augmentation:
streamlit run app.py
With augmented model:
streamlit run app_withAug.py

## Notes
It is recommended to perform model training on a GPU-enabled environment (Google
Colab/Kaggle).
The Streamlit application displays bounding boxes for workers and their PPE, and evaluates PPE
completeness per worker.


## Quick start (in Colab or local env with GPU)
1. Clone or unzip project.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. (Optional) Unzip dataset: `unzip dataset/extracted.zip -d dataset/`
4. Ensure dataset is in YOLO format
Dataset configuration files are provided as:
- data.yaml (original dataset)
- data_augmented.yaml (augmented dataset)
5. Train (example with ultralytics):
A. Training Without Augmentation
Use the notebook:
- 02_Training.ipynb Or run via CLI: yolo detect train --data data.yaml --epochs 100 --img 640 --batch 16
B. Training With Augmentation
Run these notebooks in order:
- 02A_Aug.ipynb — Generate augmented dataset
- 02B_Train With Aug.ipynb — Train model on augmented dataset Or training via CLI: yolo detect train --data data_augmented.yaml --epochs 10 --img 416 --batch 4
6.  Model Evaluation
Use the evaluation notebooks:
- 03_Evaluation.ipynb
- 03_Evaluation_Aug.ipynb
7. After training, place `best.pt` under `weights/` and run:
```bash
streamlit run app.py
```

## Contact / Notes
This deliverable is a scaffold that meets the Capstone Module 4 requirements:
- pipeline scripts
- model training script
- analysis-ready inference and calorie summation logic
- Streamlit integration for deployment to Streamlit Cloud

Please run training on a GPU machine (Google Colab recommended). See `README_TRAINING.md` for Colab-ready commands.
