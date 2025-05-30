# Chromosample - sample random sequences from FASTA/GFF3 files

## Overview

Chromosample is a Python script for sampling annotated sequences from FASTA files with GFF3 features. It can generate sequences from annotated genomic features and/or create random DNA sequences, producing a balanced dataset with comprehensive statistics and visualizations. This is very useful for preparing datasets consisting of sequences/annotation pairs for downstream processing tasks involving machine learning/classification.

In the broadest sense, *annotation* here is understood as Species/Biotype/Description:

* *Species* is self-explanatory: *Homo sapiens*, *Apis mellifera*, etc.

* *Biotype* is any kind of annotation carrying some kind of functionality, even in a loose sense: CDS, exon, miRNA, three prime UTRs, pseudogene, etc. However, some purely descriptive annotations are also often included in GFF3 files (e.g. 'chromosome', 'biological_region', etc.). These are explicitly ignored. You can specify the list of excluded annotations at the top of the files in the EXCLUDED_GFF_TYPES global. Chromosample is annotation-agnostic and does not come up with an explicit whitelist of allowed annotations. This flexibility is necessary to account for the many annotation pipelines for different species that come with various levels of confidence, fragmentation, etc. It is recommended for the user to explicitly check what kinds of annotations are likely to crop up in the input files and revise the blacklist accordingly.

* *Description* usually includes additional information about what a sequence does, typically a CDS. Depending on the GFF3, and because it is nontrivial for parsing to accomodate for any kind of annotation format in every GFF3 file, some nonfunctional description (accession numbers, etc.) can also be retained in the output. This is unfortunate but can be easily filtered out downstream.

Annotations can also be filtered out by confidence. Chromosample will look for keywords that hint at a high level (manually curated) or low level (predicted in sillico, etc.) of confidence regarding the relevance of any given annotation. Some keywords are specified in the CONFIDENCE_LEVELS global dictionary but again, user discretion is encouraged.

Chromosample can also generate random sequences. These will be labeled as "random" species, "random" biotype and come with no description. The length distribution will match that of the annotated sequences and the proportion can be specified in the command-line arguments (with a decimal).

## Key Features

- Samples sequences from annotated genomic features in GFF3 files
- Generates random DNA sequences
- Handles large genome files efficiently, supports gzipped inputs
- Produces a detailed markdown (optionally pdf) report with statistics and visualizations
- Supports multiple filtering and truncation strategies (sampling parts of annotated sequences for features with an outsized length bias, e.g. lncRNAs)

## Installation & requirements

1. Ensure Python 3.6+ is installed
2. Install required packages:
   ```bash
   pip install pyfaidx numpy
   ```
3. For full functionality (plots and report):
  ```bash
  python3 -m pip install matplotlib seaborn pandas pandoc
  ```

4. For a comprehensive report including metrics, logging, figures, etc. `pandoc` is required to convert the markdown file into a pdf.

## Command-line usage

```bash
python3 chromosample.py --fasta_patterns "path/to/*.fa" --gff_patterns "path/to/*gff" [other options]
```

### Main Arguments

#### Input/Output

```
--fasta_patterns  Glob pattern(s) for input FASTA files (required)
--gff_patterns  Glob pattern(s) for input GFF3 files
--file_pairs  Explicit species-FASTA-GFF triplets (repeatable)
--output_fasta  Path for output FASTA file (required)
--report_dir  Directory for report and plots (default: "sampler_report")
--temp_dir  Directory for temporary files
```

#### Sampling Parameters

```
--total_samples Total number of sequences to sample (default: 10000)
--annotated_proportion  Proportion from GFF annotations (default: 0.8)
--random_proportion Proportion of random DNA (default: 0.2)
--min_len Minimum sequence length (default: 50)
--max_len Maximum sequence length (default: 1000)
--truncation_strategy How to handle long sequences (default: "center")
```

#### GFF Filtering

```
--annotation_confidence_level Predefined confidence level for filtering
--target_gff_feature_types  Comma-separated feature types to target
--description_gff_attributes  Attributes to use for descriptions
Miscellaneous
Argument  Description
--seed  Random seed for reproducibility (default: 42)
--log_level Logging level (default: "INFO")
--pandoc_path Path to pandoc executable (default: "pandoc")
```

### Output Files


* Output FASTA file: Contains all sampled sequences with headers containing metadata

* Report directory (default: sampler_report):

* Markdown report (sampling_summary_report.md)

* PDF report (if pandoc is available)

* Various plots (PNG format) showing: length distributions, GC content and entropy, species and biotype distributions, GFF parsing statistics, contig concordance between FASTA and GFF, etc.

## Examples

Basic usage with automatic FASTA-GFF matching (multi-species):
```bash
python3 chromosample.py \
  --fasta_patterns "genomes/*.fa.gz" \
  --gff_patterns "annotations/*.gff.gz" \
  --output_fasta sampled_sequences.fa \
  --total_samples 5000
```

The FASTA/GFF pairs *must* begin with the same prefix (the first string in a filename before a dot, e.g. `h38` in `hg38.toplevel.fa.gz`) in order to be matched. Unmatched files will be discarded.

Explicit file pairs with custom filtering:
```bash
python3 chromosample.py \
  --file_pairs human hg38.fa hg38.gff \
  --file_pairs mouse mm10.fa mm10.gff \
  --output_fasta samples.fa \
  --min_len 100 --max_len 500 \
  --annotation_confidence_level high
```

Random sequences only:

```
python3 chromosample.py \
  --fasta_patterns "data/*.fa" \
  --output_fasta random_samples.fa \
  --annotated_proportion 0 --random_proportion 1
```
