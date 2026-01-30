#!/bin/bash
#SBATCH --time=72:00:00                    # Request 3 hours of runtime
#SBATCH --account=st-cdeboer-1            # Specify your allocation code
#SBATCH --job-name=fasya2tab         # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1             # Defines tasks per node for each sub-job.
#SBATCH --mem=32G                        # Request 8 GB of memory    
#SBATCH --output=%A_%a.fasya2tab.out        # Redirects standard output to unique files for each sub-job.
#SBATCH --error=%A_%a.fasya2tab.err         # Redirects standard error to unique files for each sub-job.
#SBATCH --mail-user=cazottes.emmanuel@gmail.com      # Email address for job notifications
#SBATCH --mail-type=ALL    

# Load conda environment
source ~/.bashrc
conda activate meme
# Script to convert FASTA files to tabular format with folderName_fileName as third column
# Uses seqkit fx2tab for conversion

# Create output directory if it doesn't exist
mkdir -p converted_files

# Loop through all directories containing .txt files
for dir in */; do
    # Remove trailing slash from directory name
    dir_name=$(basename "$dir")
    
    # Skip if not a directory or if it's the converted_files directory
    if [[ ! -d "$dir" ]] || [[ "$dir_name" == "converted_files" ]]; then
        continue
    fi
    
    echo "Processing directory: $dir_name"
    
    # Loop through all .txt files in current directory
    for file in "$dir"*.txt; do
        # Check if file exists (handles case where no .txt files exist)
        if [[ -f "$file" ]]; then
            # Get filename without extension
            filename=$(basename "$file" .txt)
            
            # Create combined identifier: folderName_fileName
            combined_id="${dir_name}_${filename}"
            
            # Convert FASTA to tab format and add combined identifier as third column
            seqkit fx2tab "$file" | awk -v identifier="$combined_id" 'BEGIN{OFS="\t"} {print $1, $2, identifier}' > "converted_files/${combined_id}_converted.tsv"
            
            echo "Converted: $file -> converted_files/${combined_id}_converted.tsv"
        fi
    done
done

# Optionally combine all converted files into one master file
echo "Combining all converted files into master_output.tsv..."
cat converted_files/*_converted.tsv > master_output.tsv

#Remove first column for polygraph
cut -f 2,3 master_output.tsv > master_output.noID.tsv
#Fasta file for fimo/sea
awk -v OFS="\t" '{print $1"|"$3, $2}' master_output.tsv | seqkit tab2fx > master_output.fasta

echo "Conversion complete!"
echo "Individual files saved in: converted_files/"
echo "Combined output saved as: master_output.tsv"