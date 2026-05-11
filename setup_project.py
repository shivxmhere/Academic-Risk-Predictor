import os
import json

def setup_project():
    project_name = "academic-risk-predictor"
    
    print(f"Starting setup for project: {project_name}")
    
    # 1. Create folder structure
    folders = [
        "data",
        "notebooks",
        "models",
        "app/pages",
        "utils",
        "outputs"
    ]
    
    for folder in folders:
        path = os.path.join(project_name, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    # 2. Create requirements.txt
    requirements = """pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.2
matplotlib==3.8.0
seaborn==0.13.0
streamlit==1.28.0
reportlab==4.0.6
joblib==1.3.2
"""
    req_path = os.path.join(project_name, "requirements.txt")
    with open(req_path, "w") as f:
        f.write(requirements)
    print(f"Created requirements.txt with pinned dependencies")

    # 3. Create empty placeholder files
    empty_files = [
        "app/app.py",
        "app/pages/predict.py",
        "app/pages/analytics.py",
        "app/pages/export.py",
        "utils/preprocessor.py",
        "utils/predictor.py"
    ]
    
    for file in empty_files:
        file_path = os.path.join(project_name, file)
        with open(file_path, "w") as f:
            pass
        print(f"Created empty Python file: {file_path}")

    # Create empty notebook JSON structure
    empty_notebook = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    notebooks = [
        "notebooks/01_eda.ipynb",
        "notebooks/02_preprocessing.ipynb",
        "notebooks/03_model_training.ipynb"
    ]
    
    for nb in notebooks:
        nb_path = os.path.join(project_name, nb)
        with open(nb_path, "w") as f:
            json.dump(empty_notebook, f, indent=1)
        print(f"Created empty Jupyter notebook: {nb_path}")

    # 4. Create README.md
    readme_content = """# Academic Risk Predictor

## Group Information
- **Group:** 174
- **Institution:** IIT Patna
- **Course:** CSDA 2025-26

## Description
A machine learning project designed to predict and analyze academic risks for students. This project is a Capstone Project for IIT Patna.
"""
    readme_path = os.path.join(project_name, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"Created README.md with project details")

    print("\nProject setup complete! 🎉")
    print(f"To get started, run:\ncd {project_name}")

if __name__ == "__main__":
    setup_project()
