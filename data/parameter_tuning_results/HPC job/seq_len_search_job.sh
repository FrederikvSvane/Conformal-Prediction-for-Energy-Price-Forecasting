#!/bin/bash
#===============================================================================
# Sequence Length Hyperparameter Search - DTU HPC Job Script
#===============================================================================

#-------------------------------------------------------------------------------
# BSUB Resource Requests
#-------------------------------------------------------------------------------
#BSUB -q gpuv100
#BSUB -J seq_len_search
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o seq_len_search_%J.out
#BSUB -e seq_len_search_%J.err

#-------------------------------------------------------------------------------
# Environment Setup
#-------------------------------------------------------------------------------
unset PYTHONHOME
unset PYTHONPATH

module purge
module load cuda/12.4
module load python3/3.11.9

#-------------------------------------------------------------------------------
# Job Information
#-------------------------------------------------------------------------------
echo "==============================================================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: seq_len_search"
echo "Queue: gpuv100"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"
echo "==============================================================================="
echo ""

#-------------------------------------------------------------------------------
# Project Setup
#-------------------------------------------------------------------------------
PROJECT_DIR="/zhome/18/a/187109/projects/CI_for_energy_price_forecasting"
cd $PROJECT_DIR

# Create/activate virtual environment
VENV_DIR="$PROJECT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Install/upgrade dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install pandas numpy scikit-learn tqdm --quiet

# Verify CUDA is available
echo ""
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

#-------------------------------------------------------------------------------
# Run Search
#-------------------------------------------------------------------------------
echo ""
echo "==============================================================================="
echo "Starting sequence length search..."
echo "==============================================================================="
echo ""

python src/seq_len_search.py

#-------------------------------------------------------------------------------
# Job Complete
#-------------------------------------------------------------------------------
echo ""
echo "==============================================================================="
echo "End Time: $(date)"
echo "==============================================================================="
