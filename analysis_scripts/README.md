# Analysis of sequences selected my Active Learning methods

This repository contains the scripts to run analysis of the sequences selected by Active Learning (AL) methods. All scripts are designed to run on a HPC with slurm as the job manager.  

The sequence analyses relies on two software packages : [polygraph](https://github.com/Genentech/polygraph) and the [MEME suite](https://meme-suite.org/). A conda environment is provided for both. `meme.yaml` and `polygraph.yaml` should be used to install the conda environments for those tools.

## Data management

Formating the data files to use as input for polygraph (tabular), fimo (fasta) and sea (fasta). The fasta to tab conversion relies on seqkit package that is contained in the meme conda environment.

### Formating selected sequences

`sbatch scripts/bash/fasta2tab.sh`

Concatenates the sequence files from each model architecture/AL method pairs into a single tabular file (`master_output.tsv`) containing the sequence ID (1st column), the nucleotide sequence (2nd column), and the model architecture/AL method (3rd column). Also produces a tabular file with no sequence ID for polygraph (`master_output.noID.tsv`) and a fasta file for fimo/sea (`master_output.fasta`).

Yeast data requires further trimming to remove the padding sequences added to the N80. Which is done by the `TrimmSequences.sh` script.

### Formating AL pool, Training pool and unselected sequences

`sbatch scripts/bash/formatFiles.sh pool.txt ALpool`

Produce the same set of files as for the selected sequences.

## Sequence analysis

### Di-nucleotide content

Di-nucleotide content is computed using the [polygraph](https://github.com/Genentech/polygraph) package.

`sbatch scripts/bash/diNuclContent.sh master_output.noID.tsv diNuclContent.selected.tsv`

### TF motifs scanning

For both human and yeast, motif scanning is done only on the AL pool but not on the selected sequences directly because it is too computationally intensive otherwise.

A custom background file is generated with `fasta-get-markov ALpool.fasta` and passed to FIMO.

#### Human

`sbatch scripts/bash/memeFimoScan.sh pool.fasta tf_motif/H13CORE_meme_format.meme hocomoco_ALpool background.txt`

Position weight matrices are downloaded from the [HOCOMOCO V13 database](https://hocomoco13.autosome.org/downloads_v13)

#### Yeast

Given the very large number of sequences in the yeast datasets, the files we splitted into smaller chunks before motif scanning with fimo.

`./split_fasta.sh <input_fasta_file>`

Then

`sbatch memeFimoScanChunksYeast.sh sequences_chunk_ background.txt motifs.meme output_dir` 

PWMs for yeast were downloaded from JASPAR's Fungi non-redundant set.

### TF motifs enrichment

TF motifs enrichment between the selected sequences versus the unselected sequences is performed using SEA usingthe same PWMs as above for each species.

`memeSEA.sh input.fasta control.fasta pwms.meme out_dir`