#!/bin/bash
#SBATCH --time=7-0                    # Request 3 hours of runtime
#SBATCH --account=st-cdeboer-1            # Specify your allocation code
#SBATCH --job-name=memeChunks         # Specify the job name
#SBATCH --nodes=1                       # Defines the number of nodes for each sub-job.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8             # Defines tasks per node for each sub-job.
#SBATCH --mem=144G                        # Request 8 GB of memory    
#SBATCH --output=%A_%a.memeChunks.out        # Redirects standard output to unique files for each sub-job.
#SBATCH --error=%A_%a.memeChunks.err         # Redirects standard error to unique files for each sub-job.
#SBATCH --mail-user=cazottes.emmanuel@gmail.com      # Email address for job notifications
#SBATCH --mail-type=ALL

# Load conda environment
source ~/.bashrc
conda activate meme

# Input parameters
CHUNK_PREFIX=$1     # Base name of chunk files (e.g., "sequences_chunk_")
BACKGROUND=$2       # Background file
MOTIFS=$3          # Motifs file
BASE_OUTDIR=$4     # Base output directory

# Validate input parameters
if [ $# -ne 4 ]; then
    echo "Usage: $0 <chunk_prefix> <background_file> <motifs_file> <base_output_dir>"
    echo "Example: $0 sequences_chunk_ background.txt motifs.meme output_dir"
    echo ""
    echo "This script will process all files matching: ${CHUNK_PREFIX}*.fasta"
    exit 1
fi

# Check if required files exist
if [ ! -f "$BACKGROUND" ]; then
    echo "Error: Background file '$BACKGROUND' does not exist."
    exit 1
fi

if [ ! -f "$MOTIFS" ]; then
    echo "Error: Motifs file '$MOTIFS' does not exist."
    exit 1
fi

# Set parameters
FDR=0.0001  # Set FDR low, might need to go lower when number of sequences increases

# Find all chunk files
CHUNK_FILES=($(ls ${CHUNK_PREFIX}*.fasta 2>/dev/null))

# Check if any chunk files were found
if [ ${#CHUNK_FILES[@]} -eq 0 ]; then
    echo "Error: No chunk files found matching pattern '${CHUNK_PREFIX}*.fasta'"
    echo "Make sure you've run the splitting script first and the chunk files exist in the current directory."
    exit 1
fi

echo "Found ${#CHUNK_FILES[@]} chunk files to process:"
for file in "${CHUNK_FILES[@]}"; do
    echo "  $file"
done
echo ""

# Create base output directory if it doesn't exist
mkdir -p "$BASE_OUTDIR"

# Process each chunk file
for CHUNK_FILE in "${CHUNK_FILES[@]}"; do
    echo "=== Processing $CHUNK_FILE ==="
    
    # Extract chunk identifier from filename (e.g., "00" from "sequences_chunk_00.fasta")
    CHUNK_ID=$(basename "$CHUNK_FILE" .fasta | sed "s/${CHUNK_PREFIX%_}_//")
    
    # Create output directory for this chunk
    CHUNK_OUTDIR="${BASE_OUTDIR}/chunk_${CHUNK_ID}_FDR0.01"
    
    echo "Input FASTA: $CHUNK_FILE"
    echo "Output directory: $CHUNK_OUTDIR"
    
    # Run FIMO on this chunk
    fimo --bfile ${BACKGROUND} \
         --max-strand \
         --thresh ${FDR} \
         --o ${CHUNK_OUTDIR} \
         --verbosity 1 \
         ${MOTIFS} \
         ${CHUNK_FILE}
    
    # Check if FIMO completed successfully
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $CHUNK_FILE"
        
        # Count number of significant hits if fimo.tsv exists
        if [ -f "${CHUNK_OUTDIR}/fimo.tsv" ]; then
            # Count lines minus header
            HITS=$(( $(wc -l < "${CHUNK_OUTDIR}/fimo.tsv") - 1 ))
            echo "  Found $HITS significant motif hits"
        fi
    else
        echo "✗ Error processing $CHUNK_FILE"
    fi
    
    echo ""
done

echo "=== BATCH PROCESSING COMPLETE ==="
echo "Processed ${#CHUNK_FILES[@]} chunk files"
echo "Results are stored in subdirectories under: $BASE_OUTDIR"

conda deactivate