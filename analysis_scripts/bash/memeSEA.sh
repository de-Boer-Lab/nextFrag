#!/bin/bash
#SBATCH --time=7-0                    # Request 3 hours of runtime
#SBATCH --account=st-cdeboer-1            # Specify your allocation code
#SBATCH --job-name=sea         # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8             # Defines tasks per node for each sub-job.
#SBATCH --mem=500G                        # Request 8 GB of memory    
#SBATCH --output=%A_%a.sea.out        # Redirects standard output to unique files for each sub-job.
#SBATCH --error=%A_%a.sea.err         # Redirects standard error to unique files for each sub-job.
#SBATCH --mail-user=cazottes.emmanuel@gmail.com      # Email address for job notifications
#SBATCH --mail-type=ALL

# Load conda environment
source ~/.bashrc
conda activate meme

INFASTA=$1
CONTROLFASTA=$2
#BACKGROUND=$3
MOTIFS=$3
OUTDIR=$4

FDR=0.01 #Set FDR low, might need to go lower when nnumber of sequences increases

SEA="/arc/project/st-cdeboer-1/emmanuel/meme-5.5.8/bin/sea"

DIRNAME=$(echo ${OUTDIR}\_FDR${FDR})

echo "Scanning for motif enrichment"

${SEA} --p ${INFASTA} \
--m ${MOTIFS} \
--n ${CONTROLFASTA} \
--thresh ${FDR} \
--qvalue \
--verbosity 1 \
--noseqs \
--o ${DIRNAME}  

conda deactivate

#--bfile ${BACKGROUND} \