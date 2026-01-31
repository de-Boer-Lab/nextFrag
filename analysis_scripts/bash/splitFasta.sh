#!/bin/bash
#SBATCH --time=72:00:00                    # Request 3 hours of runtime
#SBATCH --account=st-cdeboer-1            # Specify your allocation code
#SBATCH --job-name=splitFasta         # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8             # Defines tasks per node for each sub-job.
#SBATCH --mem=16G                        # Request 8 GB of memory    
#SBATCH --output=%A_%a.splitFasta.out        # Redirects standard output to unique files for each sub-job.
#SBATCH --error=%A_%a.splitFasta.err         # Redirects standard error to unique files for each sub-job.
#SBATCH --mail-user=cazottes.emmanuel@gmail.com      # Email address for job notifications
#SBATCH --mail-type=ALL
# FASTA File Splitter Script
# Usage: ./split_fasta.sh <input_fasta_file>

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_fasta_file>"
    echo "Example: $0 sequences.fasta"
    exit 1
fi

# Input file
input_file="$1"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' does not exist."
    exit 1
fi

# Check if input file is readable
if [ ! -r "$input_file" ]; then
    echo "Error: File '$input_file' is not readable."
    exit 1
fi

# Count total lines in the file
echo "Counting lines in '$input_file'..."
total_lines=$(wc -l < "$input_file")
echo "Total lines: $total_lines"

# Define chunk size
chunk_size=1000000

# Calculate number of chunks needed
num_chunks=$(( (total_lines + chunk_size - 1) / chunk_size ))
echo "Will create $num_chunks chunk(s) of up to $chunk_size lines each"

# Get base filename without extension for output naming
base_name=$(basename "$input_file" .fasta)
if [ "$base_name" = "$(basename "$input_file")" ]; then
    # If .fasta extension wasn't found, try other common extensions
    base_name=$(basename "$input_file" .fa)
    if [ "$base_name" = "$(basename "$input_file")" ]; then
        base_name=$(basename "$input_file" .fas)
        if [ "$base_name" = "$(basename "$input_file")" ]; then
            # No common FASTA extension found, use full filename
            base_name=$(basename "$input_file")
        fi
    fi
fi

echo "Splitting file into chunks..."

# Split the file using the split command
# -l specifies lines per chunk
# -d uses numeric suffixes instead of alphabetic
# --additional-suffix adds .fasta to each chunk
split -l "$chunk_size" -d --additional-suffix=.fasta "$input_file" "${base_name}_chunk_"

# Count and display the created chunks
chunk_count=$(ls "${base_name}_chunk_"*.fasta 2>/dev/null | wc -l)

if [ "$chunk_count" -gt 0 ]; then
    echo "Successfully created $chunk_count chunk files:"
    
    # Display information about each chunk
    for chunk_file in "${base_name}_chunk_"*.fasta; do
        if [ -f "$chunk_file" ]; then
            chunk_lines=$(wc -l < "$chunk_file")
            echo "  $chunk_file: $chunk_lines lines"
        fi
    done
    
    echo "Splitting completed successfully!"
else
    echo "Error: No chunk files were created."
    exit 1
fi