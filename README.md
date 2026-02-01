# Flight Delay Prediction – Tunisair

## Project Overview

This project focuses on predicting the estimated duration of flight delays for the Tunisian airline Tunisair using machine learning techniques. The goal is to support operational planning, reduce inefficiencies, and minimize financial and customer impact caused by delays.

---

## Business Context

Tunisair is the national airline of Tunisia, founded in 1948, operating international flights across four continents with its main hub at Tunis–Carthage International Airport.

---

## Objectives

- Predict flight delay duration using machine learning  
- Build a quick baseline model  
- Handle potential data imbalance  
- Compare at least three ML algorithms  
- Select appropriate evaluation metrics  
- Provide business-oriented insights  

---

## Deliverables

1. Slide deck (PDF) for non-technical stakeholders (15 minutes)  
2. PEP8-compliant Jupyter notebook for technical audiences  
3. Optional Python script for training and running models via CLI  
4. One slide outlining a potential data product  

---

## Project Structure

```text
.
├── data/                 # Data used for the Project
├── Models/               # Trained models
├── Models/Tests/         # Zindi submission test files
├── Notebooks_Personal/   # EDA and modeling notebooks
├── Utils/                # Utility functions for EDA and pipelines
├── requirements.txt
└── README.md
```

## Setup
The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute the Diabetes Challenge notebooks.

*Note: If there are errors during environment setup, try removing the versions from the failing packages in the requirements file. M1 shizzle.*

### **`macOS`** type the following commands : 

- We have also added a [Makefile](Makefile) which has the recipe called 'setup' which will run all the commands for setting up the environment.
Feel free to check and use if you are tired of copy pasting so many commands.

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```
Or ....
- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```
  
