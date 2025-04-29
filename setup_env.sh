#!/bin/bash

# Set environment name
ENV_NAME="RLEnv"

# Check if the environment exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
    echo "Do you wish to delete and rebuild this environment? (y/n)"
    read answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        conda remove -y -n "$ENV_NAME" --all
        echo "Environment destroyed"
        echo "Creating conda environment '$ENV_NAME'..."
        conda create -y -n "$ENV_NAME" python=3.8
        echo "Environment created."
    else
        echo "Continuing with existing environment"
    fi
else
    # Create a fresh environment
    # Use Python 3.8 which has better compatibility with Ray on M1
    conda create -y -n "$ENV_NAME" python=3.8
    echo "Environment created."
fi

# Activate the environment
echo "Activating environment '$ENV_NAME'..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Verify activation worked
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "Error: Failed to activate environment. Please run: conda activate $ENV_NAME"
    exit 1
fi

echo "Installing dependencies in environment '$ENV_NAME'..."

# Install primary conda packages
conda install -y -c conda-forge networkx matplotlib numpy scipy
conda install -y -c conda-forge gymnasium
conda install -y -c pytorch pytorch
conda install -y -c conda-forge tqdm pandas seaborn
conda install -y -c conda-forge ipykernel jupyter
conda install -y -c conda-forge python-graphviz pydot
conda install -y -c conda-forge tensorboard

# Install cmake (needed for Ray compilation)
pip install cmake

# Install Ray with RLlib using pip
echo "Installing Ray RLlib..."
pip install "ray[rllib]"

# Register the kernel with Jupyter
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo "Installation complete! Environment '$ENV_NAME' is ready to use."
echo "To activate this environment in new terminals, run: conda activate $ENV_NAME"