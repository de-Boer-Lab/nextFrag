#!/bin/bash
#SBATCH --time=72:00:00                    # Request 3 hours of runtime
#SBATCH --account=st-cdeboer-1            # Specify your allocation code
#SBATCH --job-name=TimmSeq         # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8             # Defines tasks per node for each sub-job.
#SBATCH --mem=8G                        # Request 8 GB of memory    
#SBATCH --output=%A_%a.TimmSeq.out        # Redirects standard output to unique files for each sub-job.
#SBATCH --error=%A_%a.TimmSeq.err         # Redirects standard error to unique files for each sub-job.
#SBATCH --mail-user=cazottes.emmanuel@gmail.com      # Email address for job notifications
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate meme

#This script trimms the padding added to each sequence of the yeast dataset and produces the files that will be used for downstream analysis

INFILE=$1

FILENAME=$(basename ${INFILE} .tsv)
#Trim the sequences
awk -v OFS="\t" '{print $1, substr($2, 58, 80), $3}' ${INFILE} > ${FILENAME}.trim.tsv

#Remove the ID column for polygraph input
cut -f 2,3 ${FILENAME}.trim.tsv > ${FILENAME}.trim.noID.tsv

#Fasta sequence
awk -v OFS="\t" '{print $1"|"$3, $2}' ${FILENAME}.trim.tsv | seqkit tab2fx > ${FILENAME}.trim.fasta

conda deactivate 
