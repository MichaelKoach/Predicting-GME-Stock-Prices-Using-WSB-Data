#!/bin/bash
# Run this from the root of your cloned repository

mkdir -p notebooks
mkdir -p data/raw
mkdir -p data/processed

# Keep empty folders tracked by git
touch data/raw/.gitkeep
touch data/processed/.gitkeep

echo "Folder structure created."
