# SEM-abide-networks

This master's thesis project explores brain connectivity networks using Structural Equation Modeling (SEM) and network analysis, providing evidence that the graph spectral radius is a useful proxy to use as feature in brain imaging analysis, particularly in the context of autism spectrum disorder.

## Project Structure

The repository is organized as follows:

- `notebooks/`: Contains Jupyter Notebooks with the logical sequence of the project's analysis:
  - `01_preprocessamento.ipynb`: Raw data preprocessing.
  - `02_analise_exploratoria.ipynb`: Exploratory Data Analysis (EDA).
  - `03_modelgem.ipynb`: Model building and fitting.
  - `04_selacao_modelo.ipynb`: Model comparison and selection.
  - `05_simulacoes.ipynb`: Simulations based on the chosen models.
  - `06_plots.ipynb`: Generation of final plots and visualizations.
- `dados/`: Directory for storing the data used and generated during the analyses.
- `utils/`: Auxiliary files, including the data dictionary (ABIDE legend) and DAG formulas.
- `environment.yml`: Conda environment configuration file to ensure project reproducibility.

## Prerequisites and Installation

To run the notebooks and reproduce the development environment, you will need [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda) installed.

To create and activate the virtual environment with the project dependencies (`pandas`, `networkx`, `semopy`, `nilearn`, etc.), run the following commands in your terminal:

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the virtual environment named "redes_env"
conda activate redes_env
```

## How to Use

With the environment activated, start Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and run the scripts in numbered order to reproduce the complete analysis pipeline.
