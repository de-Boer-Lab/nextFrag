#!/bin/bash
#SBATCH --time=72:00:00                    # Request 3 hours of runtime
#SBATCH --account=st-cdeboer-1            # Specify your allocation code
#SBATCH --job-name=tab2fasta         # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1             # Defines tasks per node for each sub-job.
#SBATCH --mem=32G                        # Request 8 GB of memory    
#SBATCH --output=%A_%a.tab2fasta.out        # Redirects standard output to unique files for each sub-job.
#SBATCH --error=%A_%a.tab2fasta.err         # Redirects standard error to unique files for each sub-job.
#SBATCH --mail-user=cazottes.emmanuel@gmail.com      # Email address for job notifications
#SBATCH --mail-type=ALL    

# Load conda environment
source ~/.bashrc
conda activate meme

#This script convert the tab sequence format orginally used with polygraph for motif scanning with the meme suite because there is a loss of sequence information 
# with polygraph

INFASTA=$1
GROUP=$2

FILENAME=$(basename ${INFASTA} .txt)

seqkit fx2tab ${INFASTA} | awk -v OFS="\t" -v group=${GROUP} '{print $1,$2,group}'> ${FILENAME}.tsv
cut -f 2,3 ${FILENAME}.tsv > ${FILENAME}.noID.tsv
#Add the metadata (e.g. the model architecture, the AL selection method and the seed number) of each sequence to the sequence name  of the fasta file
awk -v OFS="\t" '{print $1"|"$3, $2}' ${FILENAME}.tsv | seqkit tab2fx > ${FILENAME}.fasta

conda deactivate