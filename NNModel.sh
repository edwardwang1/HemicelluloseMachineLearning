#!/bin/bash
#SBATCH --gres=gpu:3        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-8:00      # time (DD-HH:MM)
#SBATCH --output=NNModel-%N-%j.out  # %N for node name, %j for jobID

module load python/3.6
module load scipy-stack
module load cuda cudnn 
source $HOME/hemicellulose_project/bin/activate

python ./NNModel.py
