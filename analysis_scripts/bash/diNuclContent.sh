#!/bin/bash
#SBATCH --time=72:00:00                    # Request 3 hours of runtime
#SBATCH --account=st-cdeboer-1            # Specify your allocation code
#SBATCH --job-name=diNucl         # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8             # Defines tasks per node for each sub-job.
#SBATCH --mem=96G                        # Request 8 GB of memory    
#SBATCH --output=%A_%a.diNucl.out        # Redirects standard output to unique files for each sub-job.
#SBATCH --error=%A_%a.diNucl.err         # Redirects standard error to unique files for each sub-job.
#SBATCH --mail-user=cazottes.emmanuel@gmail.com      # Email address for job notifications
#SBATCH --mail-type=ALL

#For Numba and matplotlib caching 

export MPLCONFIGDIR=/scratch/st-cdeboer-1/emmanuel/cache/matplotlib
export NUMBA_CACHE_DIR=/scratch/st-cdeboer-1/emmanuel/cache/numba

source ~/.bashrc
conda activate /arc/project/st-cdeboer-1/emmanuel/miniconda3/envs/polygraph

PYSCRIPT="/scratch/st-cdeboer-1/emmanuel/al_selected_seqs/scripts/python/getDiNuclContent.py"

INTAB=$1
OUTFILE=$2

python ${PYSCRIPT} ${INTAB} ${OUTFILE}