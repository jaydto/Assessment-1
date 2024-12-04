# Random Users Project

This project fetches random user data from an API, stores it in an SQLite database, and performs analysis on the data, including finding common properties (such as gender) and identifying similarities between users.

## Python version
- Current version used in environment is `Python 3.8.18` 

## Git 

- `feature/weak-strong-relationship` is an alteration of weak , strong relationship mapping using rapid fuzzy for optimization while `master` contains various visualizations using fuzzy 





## Features

- **Data Fetching:** Retrieves random user data from an API.
- **Database Management:** Uses SQLite for data storage to keep everything simple and in a single file.
- **Data Analysis:** Finds similar users based on properties such as gender and username similarity.
- **Notebook Generation:** Produces an interactive Jupyter notebook with visualizations and analysis.
- **Logging:** Records all operations for debugging and traceability.

---


## Setup
1. Clone the Repository
    ```
    git clone git@github.com:jaydto/Assessment-1.git
    ```

2. Create a virtual environment:
    ```
    python3 -m venv venv
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Run the scripts:
    - Fetch and save random users: `python scripts/fetch_users.py`
    - Load data into SQLite and perform analysis: `python -m scripts.load_data`

## Analysis Output
- The output of the analysis is stored in a Jupyter notebook:

- File Location: `notebooks/user_analysis.ipynb`

### Visualizations Included:
- Distribution of username similarity scores.
- Gender match distribution.

#### To view the notebook:

- Install Jupyter Notebook:

`pip install notebook`
- Open the notebook:

`jupyter notebook notebooks/user_analysis.ipynb`

## Project Structure
### The project is organized as follows: 
```
.
├── LICENSE
├── README.md
├── data
│   └── random_users.csv
├── db
│   └── schema.py
├── logs
│   └── app.log  # Log file saved here
├── notebooks
│   └── user_analysis.ipynb
├── requirements.txt
└── scripts
    ├── fetch_users.py
    └── load_data.py

    
```

## Sql 

- Nb//: using `sqlite` for ease of use and to ensure you can have everything under one file 

## Installing make file 
- **On Linux (Debian/Ubuntu):**
  To install `make` on a Debian-based Linux distribution, run the following commands:
  
```
sudo apt update
sudo apt install make

```
- **On Macos**:
```
brew install make

```
- **On windows**

1. Windows Subsystem for Linux (WSL): If you are using WSL, you can install make inside your WSL environment using the Linux commands above.
2. Git Bash: If you have Git for Windows installed, you can use the make command through Git Bash. Git Bash comes with make and many other Unix utilities.
3. Install through MinGW: Alternatively, you can install make via MinGW or other Windows package managers.



## Using the Makefile
### The project includes a Makefile for simplified execution:
#### General Usage

```
make clean 
make run_analysis
make run_notebook_view

```
### Independent Usage

```
make fetch_users
make clean_notebooks
make load_data
make run_notebook
```

## Running Notebooks

- `Nb:// make run_notebook` will create a separate `.ipynb` that has been run and show results.
- `Nb:// make run_notebook_view` will take you to the notebook to see it.
- `Nb:// make clean_notebooks` will clean your notebook records.

## Run Analysis

- `Nb:// Run Analysis` runs both fetch and load data .

## Cleaning

- `make clean` will clean up temporary files and reset the environment.
