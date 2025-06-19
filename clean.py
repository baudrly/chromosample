#!/usr/bin/env python3
"""
Smart Dataset Cleaner for Genomic Sequences

Cleans up genomic sequence datasets by:
- Standardizing biotype names using comprehensive ontology mapping
- Fixing case issues and header formatting
- Removing rare species/biotypes intelligently
- Filtering low-quality sequences (N-rich)
- Removing unknown/broken entries with smart detection
- Preventing biotype consolidation that would create dominance
"""

import argparse
import re
import sys
import gzip
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
import logging
import math
import numpy as np

# Comprehensive biotype ontology mapping
BIOTYPE_ONTOLOGY = {
    # Protein coding variants
    "protein_coding": "protein_coding",
    "protein_coding_gene": "protein_coding",
    "cds": "protein_coding",
    "coding": "protein_coding",
    "mrna": "protein_coding",
    "messenger_rna": "protein_coding",
    "processed_transcript": "protein_coding",
    # Long non-coding RNA family
    "lncrna": "lncRNA",
    "lnc_rna": "lncRNA",
    "long_noncoding_rna": "lncRNA",
    "lincrna": "lncRNA",
    "antisense": "lncRNA",
    "antisense_rna": "lncRNA",
    "sense_intronic": "lncRNA",
    "sense_overlapping": "lncRNA",
    "3prime_overlapping_ncrna": "lncRNA",
    "bidirectional_promoter_lncrna": "lncRNA",
    "macro_lncrna": "lncRNA",
    # Small RNA family - keep specific where functionally distinct
    "mirna": "miRNA",
    "micro_rna": "miRNA",
    "mir": "miRNA",
    "pirna": "piRNA",
    "piwi_interacting_rna": "piRNA",
    "sirna": "siRNA",
    "small_interfering_rna": "siRNA",
    # Ribosomal RNA - keep separate due to functional importance
    "rrna": "rRNA",
    "ribosomal_rna": "rRNA",
    "rdna": "rRNA",
    "5s_rrna": "rRNA",
    "5_8s_rrna": "rRNA",
    "18s_rrna": "rRNA",
    "28s_rrna": "rRNA",
    "45s_pre_rrna": "rRNA",
    # Transfer RNA - functionally important
    "trna": "tRNA",
    "transfer_rna": "tRNA",
    "trna_gene": "tRNA",
    "mt_trna": "tRNA",  # mitochondrial tRNA
    # spliceosomal RNAs - keep separate due to functional importance
    "snrna": "snRNA",
    "small_nuclear_rna": "snRNA",
    "u1": "snRNA",
    "u2": "snRNA",
    "u4": "snRNA",
    "u5": "snRNA",
    "u6": "snRNA",
    "u11": "snRNA",
    "u12": "snRNA",
    "snorna": "snoRNA",
    "small_nucleolar_rna": "snoRNA",
    "snord": "snoRNA",
    "snora": "snoRNA",
    # Pseudogene family - consolidate functionally similar
    "pseudogene": "pseudogene",
    "processed_pseudogene": "pseudogene",
    "unprocessed_pseudogene": "pseudogene",
    "unitary_pseudogene": "pseudogene",
    "polymorphic_pseudogene": "pseudogene",
    "ig_pseudogene": "pseudogene",  # immunoglobulin pseudogenes
    "tr_pseudogene": "pseudogene",  # T-cell receptor pseudogenes
    "ig_c_pseudogene": "pseudogene",
    "ig_d_pseudogene": "pseudogene",
    "ig_j_pseudogene": "pseudogene",
    "ig_v_pseudogene": "pseudogene",
    "tr_c_pseudogene": "pseudogene",
    "tr_d_pseudogene": "pseudogene",
    "tr_j_pseudogene": "pseudogene",
    "tr_v_pseudogene": "pseudogene",
    "transcribed_processed_pseudogene": "pseudogene",
    "transcribed_unprocessed_pseudogene": "pseudogene",
    "transcribed_unitary_pseudogene": "pseudogene",
    "translated_processed_pseudogene": "pseudogene",
    "translated_unprocessed_pseudogene": "pseudogene",
    "rRNA_pseudogene": "pseudogene",
    # Immunoglobulin genes - keep separate due to medical importance
    "ig_c_gene": "immunoglobulin",
    "ig_d_gene": "immunoglobulin",
    "ig_j_gene": "immunoglobulin",
    "ig_v_gene": "immunoglobulin",
    "immunoglobulin": "immunoglobulin",
    "antibody": "immunoglobulin",
    # T-cell receptor genes - keep separate
    "tr_c_gene": "t_cell_receptor",
    "tr_d_gene": "t_cell_receptor",
    "tr_j_gene": "t_cell_receptor",
    "tr_v_gene": "t_cell_receptor",
    "tcr": "t_cell_receptor",
    # Structural elements - consolidate non-functional annotations
    # "exon": "structural",
    # "intron": "structural",
    "cds": "protein_coding",  # override - CDS should be protein coding
    "5_prime_utr": "UTR",
    "3_prime_utr": "UTR",
    "five_prime_utr": "UTR",
    "three_prime_utr": "UTR",
    "utr": "UTR",
    "utr5": "UTR",
    "utr3": "UTR",
    # Regulatory elements - consolidate rare regulatory annotations
    # "enhancer": "regulatory",
    # "silencer": "regulatory",
    # "promoter": "regulatory",
    "regulatory_region": "regulatory",
    "transcriptional_cis_regulatory_region": "regulatory",
    "cpg_island": "regulatory",
    "caat_signal": "regulatory",
    "tata_box": "regulatory",
    "minus_10_signal": "regulatory",
    "minus_35_signal": "regulatory",
    # Repeat elements - consolidate
    "transposable_element": "transposon",  # "repeat",
    "retrotransposon": "transposon",  # "repeat",
    "dna_transposon": "transposon",  # "repeat",
    "line": "transposon",  # "repeat",
    "sine": "transposon",  # , "repeat",
    "ltr": "transponson",  # , "repeat",
    "tandem_repeat": "repeat",
    "satellite_dna": "repeat",
    "centromeric_repeat": "repeat",
    # Miscellaneous ncRNA - consolidate very rare types
    "scrna": "other_ncRNA",
    "small_cytoplasmic_rna": "other_ncRNA",
    "rnase_p_rna": "other_ncRNA",
    "rnase_mrp_rna": "other_ncRNA",
    "telomerase_rna": "other_ncRNA",
    "vault_rna": "other_ncRNA",
    "y_rna": "other_ncRNA",
    "guide_rna": "other_ncRNA",
    "ribozyme": "other_ncRNA",
    "hammerhead_ribozyme": "other_ncRNA",
    "autocatalytically_spliced_intron": "other_ncRNA",
    # Nonsense mediated decay
    "non_stop_decay": "NMD_target",
    "nonsense_mediated_decay": "NMD_target",
    "nmd_transcript_variant": "NMD_target",
    "protein_coding_lof": "NMD_target",
    # Low confidence / predictions - consolidate
    "tec": "low_confidence",  # To be Experimentally Confirmed
    "novel_transcript": "low_confidence",
    "predicted_gene": "low_confidence",
    "ab_initio_prediction": "low_confidence",
    "computational_prediction": "low_confidence",
}

# Species name standardization patterns
SPECIES_STANDARDIZATION = {
    # Common organism name fixes
    "homo_sapiens": "Homo_sapiens",
    "mus_musculus": "Mus_musculus",
    "drosophila_melanogaster": "Drosophila_melanogaster",
    "caenorhabditis_elegans": "Caenorhabditis_elegans",
    "saccharomyces_cerevisiae": "Saccharomyces_cerevisiae",
    "arabidopsis_thaliana": "Arabidopsis_thaliana",
    "escherichia_coli": "Escherichia_coli",
    "bacillus_subtilis": "Bacillus_subtilis",
    # Handle common abbreviations
    "chm13v2": "Homo_sapiens",
    "h_sapiens": "Homo_sapiens",
    "m_musculus": "Mus_musculus",
    "d_melanogaster": "Drosophila_melanogaster",
    "c_elegans": "Caenorhabditis_elegans",
    "s_cerevisiae": "Saccharomyces_cerevisiae",
    "a_thaliana": "Arabidopsis_thaliana",
    "e_coli": "Escherichia_coli",
}

UNKNOWN_SPECIES_PATTERNS = {
    # Only obvious unknowns
    "unknown", "unknown_species", "unidentified", "unassigned", "na", "n/a", 
    "Unknown_species",
    "null", "none", "missing", "blank", "empty", "undefined", "unspecified",
    
    # Clear non-biological terms
    "test", "example", "demo", "temp", "tmp", "placeholder", "dummy",
    "file", "database", "db", "fasta", "seq", "header",
    
    # Single characters or obvious placeholders
    "x", "y", "z", "unk", "xxx", "yyy", "zzz", "aaa", "bbb", "ccc",
    "a", "b", "c", "1", "2", "3", "0",
}

UNKNOWN_BIOTYPE_PATTERNS = {
    # Only obvious unknowns
    "unknown", "unclassified", "unidentified", "unassigned", "undefined",
    "unspecified", "undetermined", "unclear", "ambiguous", "na", "n/a",
    "null", "none", "missing", "blank", "empty",
    
    # Clear placeholders
    "temp", "tmp", "placeholder", "dummy", "test", "example", "demo",
    "xxx", "yyy", "zzz", "aaa", "bbb", "tbd", "todo",
    
    # Single characters
    "x", "y", "z", "a", "b", "c", "1", "2", "3", "0",
}


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def calculate_n_content(sequence: str) -> float:
    """Calculate percentage of N bases in sequence."""
    if not sequence:
        return 100.0
    n_count = sequence.upper().count("N")
    return (n_count / len(sequence)) * 100.0


def is_valid_dna_sequence(sequence: str, min_valid_bases: float = 0.8) -> bool:
    """Check if sequence contains valid DNA bases."""
    if not sequence:
        return False

    valid_bases = set("ATCGN")
    valid_count = sum(1 for base in sequence.upper() if base in valid_bases)
    valid_fraction = valid_count / len(sequence)

    return valid_fraction >= min_valid_bases


def detect_broken_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Detect if a sequence is broken or invalid.

    Returns:
        (is_broken, reason)
    """
    if not sequence or len(sequence.strip()) == 0:
        return True, "empty_sequence"

    sequence = sequence.upper().strip()

    # Check for minimum length
    if len(sequence) < 10:
        return True, "too_short"

    # Check for excessive repetition
    most_common_base = max("ATCGN", key=sequence.count)
    if sequence.count(most_common_base) / len(sequence) > 0.95:
        return True, f"excessive_{most_common_base}_content"

    # Check for invalid characters
    valid_bases = set("ATCGN")
    invalid_chars = set(sequence) - valid_bases
    if invalid_chars:
        return True, f"invalid_characters_{len(invalid_chars)}"

    # Check for suspicious patterns
    if re.search(r"(.)\1{20,}", sequence):  # 20+ identical bases in a row
        return True, "long_homopolymer"

    # Check for only N's (or nearly all N's)
    if sequence.count("N") / len(sequence) > 0.9:
        return True, "mostly_Ns"

    return False, "valid"


def is_unknown_species(species: str) -> bool:
    """Check if species name indicates unknown/broken entry - fixed to be less aggressive."""
    if not species:
        return True
    
    species_clean = species.lower().strip().replace('_', ' ').replace('-', ' ')
    
    # Only check for exact matches of clearly unknown terms
    exact_unknown_terms = {
        "unknown", "unknown_species", "unidentified", "unassigned", "na", "n/a", 
        "null", "none", "missing", "blank", "empty", "undefined", "unspecified",
        "species", "organism", "sample", "sequence", "entry", "record", "data",
        "test", "example", "demo", "temp", "tmp", "placeholder", "dummy",
        "x", "y", "z", "unk", "xxx", "yyy", "zzz", "aaa", "bbb", "ccc",
        "a", "b", "c", "1", "2", "3", "0"
    }
    
    # Check for exact matches only
    if species_clean in exact_unknown_terms:
        return True
    
    # Check for very short names (likely not real species)
    if len(species_clean.replace(' ', '')) <= 2:
        return True
    
    # Check if it's just numbers
    if species_clean.replace(' ', '').replace('_', '').isdigit():
        return True
    
    # Check for obvious accession patterns (but be more specific)
    species_no_spaces = species.replace('_', '').replace(' ', '')
    if re.match(r'^[A-Z]{1,3}[0-9]{6,}$', species_no_spaces):
        return True
    
    # Check for file path remnants
    if any(char in species for char in ['/', '\\', '.fasta', '.fa', '.gz']):
        return True
    
    # Don't flag valid binomial nomenclature or reasonable names
    return False

def is_unknown_biotype(biotype: str) -> bool:
    """Check if biotype indicates unknown/broken entry - fixed to be less aggressive."""
    if not biotype:
        return True
    
    biotype_clean = biotype.lower().strip().replace('_', ' ').replace('-', ' ')
    
    # Only exact matches for clearly unknown/placeholder terms
    exact_unknown_terms = {
        "unknown", "unclassified", "unidentified", "unassigned", "undefined",
        "unspecified", "undetermined", "unclear", "ambiguous", "na", "n/a",
        "null", "none", "missing", "blank", "empty", 
        "other", "misc", "miscellaneous", "various", "mixed", "general",
        "temp", "tmp", "placeholder", "dummy", "test", "example", "demo",
        "xxx", "yyy", "zzz", "aaa", "bbb", "tbd", "todo",
        "x", "y", "z", "a", "b", "c", "1", "2", "3", "0"
    }
    
    # Check for exact matches
    if biotype_clean in exact_unknown_terms:
        return True
    
    # Check for very short biotypes (likely not meaningful)
    if len(biotype_clean.replace(' ', '')) <= 1:
        return True
    
    # Check if it's just numbers
    if biotype_clean.replace(' ', '').replace('_', '').isdigit():
        return True
    
    # Don't flag legitimate biotype terms, even if generic
    # Allow terms like "gene", "transcript", "rna", "dna" as they may be legitimate
    return False


def is_broken_header(header: str) -> Tuple[bool, str]:
    """
    Detect if a FASTA header is broken or malformed.

    Returns:
        (is_broken, reason)
    """
    if not header:
        return True, "empty_header"

    header = header.strip()

    # Must start with >
    if not header.startswith(">"):
        return True, "missing_fasta_prefix"

    content = header[1:].strip()
    if not content:
        return True, "empty_after_prefix"

    # Check for common broken patterns
    if content.lower() in ["sequence", "header", "fasta", "entry", "record"]:
        return True, "generic_placeholder"

    # Check for suspicious repetition
    if len(set(content.replace(" ", "").replace("|", ""))) <= 2:
        return True, "excessive_repetition"

    # Check for binary/corrupt data indicators
    if any(ord(char) < 32 or ord(char) > 126 for char in content if char not in "\t\n"):
        return True, "non_printable_characters"

    return False, "valid"


def parse_fasta_header(header: str) -> Tuple[str, str, str]:
    """
    Smart parsing of FASTA headers - improved to better handle real data.
    """
    if not header:
        return "unknown_species", "unclassified", ""
    
    header = header.lstrip('>')
    
    # Try pipe-separated format first
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 3:
            species = parts[0].strip() if parts[0].strip() else "unknown_species"
            biotype = parts[1].strip() if parts[1].strip() else "unclassified"
            description = '|'.join(parts[2:]).strip()
            return species, biotype, description
        elif len(parts) == 2:
            species = parts[0].strip() if parts[0].strip() else "unknown_species"
            second_part = parts[1].strip()
            
            # Check if second part looks like a biotype
            biotype_indicators = [
                'rna', 'gene', 'coding', 'pseudogene', 'protein', 'transcript',
                'mrna', 'trna', 'rrna', 'snrna', 'snorna', 'mirna', 'lncrna',
                'cds', 'exon', 'intron', 'utr'
            ]
            
            if any(indicator in second_part.lower() for indicator in biotype_indicators):
                return species, second_part, ""
            else:
                return species, "unclassified", second_part
    
    # Try space-separated format
    parts = header.split()
    if not parts:
        return "unknown_species", "unclassified", ""
    
    if len(parts) >= 3:
        # Look for genus_species pattern
        for i in range(len(parts)-1):
            # Check if we have a reasonable genus_species pattern
            genus_candidate = parts[i]
            species_candidate = parts[i+1]
            
            # More flexible species detection
            if (len(genus_candidate) >= 3 and len(species_candidate) >= 3 and
                genus_candidate[0].isupper() and species_candidate.islower()):
                
                species = f"{genus_candidate}_{species_candidate}"
                remaining = parts[i+2:]
                
                # Look for biotype in remaining parts
                biotype = "unclassified"
                desc_parts = []
                
                biotype_indicators = [
                    'rna', 'gene', 'coding', 'pseudogene', 'protein', 'transcript',
                    'mrna', 'trna', 'rrna', 'snrna', 'snorna', 'mirna', 'lncrna'
                ]
                
                for part in remaining:
                    part_lower = part.lower()
                    if any(indicator in part_lower for indicator in biotype_indicators):
                        if biotype == "unclassified":
                            biotype = part
                        else:
                            desc_parts.append(part)
                    else:
                        desc_parts.append(part)
                
                description = ' '.join(desc_parts)
                return species, biotype, description
    
    # Fallback parsing - be more generous
    if len(parts) >= 2:
        # First part as species, second as biotype/description
        species = parts[0] if parts[0] else "unknown_species"
        
        # If second part looks like biotype, use it; otherwise treat as description
        second_part = parts[1]
        biotype_indicators = [
            'rna', 'gene', 'coding', 'pseudogene', 'protein', 'transcript'
        ]
        
        if any(indicator in second_part.lower() for indicator in biotype_indicators):
            biotype = second_part
            description = ' '.join(parts[2:]) if len(parts) > 2 else ""
        else:
            biotype = "unclassified"
            description = ' '.join(parts[1:])
        
        return species, biotype, description
    
    elif len(parts) == 1:
        # Single part - try to extract meaningful information
        single_part = parts[0]
        
        # If it contains underscores, might be genus_species
        if '_' in single_part:
            name_parts = single_part.split('_')
            if len(name_parts) >= 2:
                return single_part, "unclassified", ""
        
        return single_part, "unclassified", ""
    
    return "unknown_species", "unclassified", ""

    # Fallback: treat first part as species, second as biotype, rest as description
    if len(parts) >= 2:
        return parts[0], parts[1], " ".join(parts[2:])
    elif len(parts) == 1:
        return parts[0], "unclassified", ""
    else:
        return "unknown_species", "unclassified", ""


def standardize_biotype(biotype: str, ontology: Dict[str, str]) -> str:
    """Standardize biotype name using ontology mapping."""
    if not biotype:
        return "unclassified"

    # Clean and normalize
    biotype_clean = biotype.lower().strip().replace(" ", "_").replace("-", "_")

    # Direct mapping
    if biotype_clean in ontology:
        return ontology[biotype_clean]

    # Partial matching for complex biotypes
    for pattern, canonical in ontology.items():
        if pattern in biotype_clean or biotype_clean in pattern:
            return canonical

    # Check for common patterns not in ontology
    if "pseudogene" in biotype_clean:
        return "pseudogene"
    elif any(term in biotype_clean for term in ["rna", "ncrna"]):
        if any(term in biotype_clean for term in ["long", "lnc"]):
            return "lncRNA"
        else:
            return "other_ncRNA"
    elif any(term in biotype_clean for term in ["coding", "protein", "mrna", "cds"]):
        return "protein_coding"

    return biotype_clean


def standardize_species(species: str, standardization: Dict[str, str]) -> str:
    """Standardize species name."""
    if not species:
        return "unknown_species"

    species_clean = species.lower().strip().replace(" ", "_")

    # Direct mapping
    if species_clean in standardization:
        return standardization[species_clean]

    # Apply proper capitalization for binomial nomenclature
    parts = species_clean.split("_")
    if len(parts) >= 2:
        return f"{parts[0].capitalize()}_{parts[1].lower()}"

    return species.replace(" ", "_")


def analyze_biotype_distribution(
    biotype_counts: Counter, target_total: int, max_dominance: float = 0.5
) -> Dict[str, str]:
    """
    Analyze biotype distribution and recommend consolidations that won't create dominance.
    """
    total_samples = sum(biotype_counts.values())
    max_allowed = int(target_total * max_dominance)

    # Start with direct ontology mapping
    consolidation_map = {}
    consolidated_counts = Counter()

    # First pass: apply ontology mapping and count
    for biotype, count in biotype_counts.items():
        mapped = standardize_biotype(biotype, BIOTYPE_ONTOLOGY)
        consolidation_map[biotype] = mapped
        consolidated_counts[mapped] += count

    # Second pass: check for dominance and split if necessary
    final_map = {}
    for original_biotype, mapped_biotype in consolidation_map.items():
        if consolidated_counts[mapped_biotype] > max_allowed:
            # If consolidation would create dominance, keep specific subtypes
            if mapped_biotype == "pseudogene":
                # Keep immunoglobulin and T-cell receptor pseudogenes separate
                if "ig_" in original_biotype.lower():
                    final_map[original_biotype] = "IG_pseudogene"
                elif "tr_" in original_biotype.lower():
                    final_map[original_biotype] = "TR_pseudogene"
                else:
                    final_map[original_biotype] = "pseudogene"
            elif mapped_biotype == "other_ncRNA":
                # Keep more specific ncRNA types
                if any(
                    term in original_biotype.lower()
                    for term in ["scrna", "vault", "y_rna"]
                ):
                    final_map[original_biotype] = original_biotype
                else:
                    final_map[original_biotype] = mapped_biotype
            else:
                final_map[original_biotype] = mapped_biotype
        else:
            final_map[original_biotype] = mapped_biotype

    return final_map


def filter_rare_categories(
    counts: Counter, min_count: int, min_proportion: float, total_samples: int
) -> Set[str]:
    """Identify categories to remove based on rarity thresholds."""
    min_threshold = max(min_count, int(total_samples * min_proportion))
    return {category for category, count in counts.items() if count < min_threshold}


class SmartDatasetCleaner:
    """Main class for intelligent dataset cleaning with unknown/broken entry detection."""

    def __init__(
        self,
        min_species_count: int = 10,
        min_biotype_count: int = 5,
        min_species_proportion: float = 0.001,
        min_biotype_proportion: float = 0.0005,
        max_n_content: float = 50.0,
        max_biotype_dominance: float = 0.5,
        preserve_rare_functional: bool = True,
        remove_unknown_species: bool = True,
        remove_unknown_biotypes: bool = True,
        remove_broken_sequences: bool = True,
        remove_broken_headers: bool = True,
        min_valid_dna_bases: float = 0.8,
    ):
        """
        Initialize cleaner with filtering parameters.

        Args:
            min_species_count: Minimum absolute count for species
            min_biotype_count: Minimum absolute count for biotype
            min_species_proportion: Minimum proportion for species
            min_biotype_proportion: Minimum proportion for biotype
            max_n_content: Maximum percentage of N bases allowed
            max_biotype_dominance: Maximum fraction any biotype can represent
            preserve_rare_functional: Keep functionally important rare biotypes
            remove_unknown_species: Remove entries with unknown species names
            remove_unknown_biotypes: Remove entries with unknown biotype names
            remove_broken_sequences: Remove broken/invalid sequences
            remove_broken_headers: Remove entries with broken headers
            min_valid_dna_bases: Minimum fraction of valid DNA bases required
        """
        self.min_species_count = min_species_count
        self.min_biotype_count = min_biotype_count
        self.min_species_proportion = min_species_proportion
        self.min_biotype_proportion = min_biotype_proportion
        self.max_n_content = max_n_content
        self.max_biotype_dominance = max_biotype_dominance
        self.preserve_rare_functional = preserve_rare_functional
        self.remove_unknown_species = remove_unknown_species
        self.remove_unknown_biotypes = remove_unknown_biotypes
        self.remove_broken_sequences = remove_broken_sequences
        self.remove_broken_headers = remove_broken_headers
        self.min_valid_dna_bases = min_valid_dna_bases

        # Functionally important biotypes to preserve even if rare
        self.functional_biotypes = {
            "miRNA",
            "tRNA",
            "rRNA",
            "snRNA",
            "snoRNA",
            "immunoglobulin",
            "t_cell_receptor",
        }

        self.logger = logging.getLogger(__name__)

    def clean_dataset(self, input_file: str, output_file: str) -> Dict[str, any]:
        """
        Main cleaning function with comprehensive unknown/broken entry detection.

        Returns statistics about the cleaning process.
        """
        self.logger.info(
            f"Starting enhanced dataset cleaning: {input_file} -> {output_file}"
        )

        # First pass: read and validate all sequences
        sequences = self._read_sequences(input_file)
        original_count = len(sequences)
        self.logger.info(f"Read {original_count} sequences")

        # Track removal reasons
        removal_stats = Counter()

        # Filter broken headers and sequences first
        valid_sequences = []
        for header, sequence in sequences:
            # Check for broken header
            if self.remove_broken_headers:
                is_broken_hdr, reason = is_broken_header(header)
                if is_broken_hdr:
                    removal_stats[f"broken_header_{reason}"] += 1
                    continue

            # Check for broken sequence
            if self.remove_broken_sequences:
                is_broken_seq, reason = detect_broken_sequence(sequence)
                if is_broken_seq:
                    removal_stats[f"broken_sequence_{reason}"] += 1
                    continue

                # Check DNA validity
                if not is_valid_dna_sequence(sequence, self.min_valid_dna_bases):
                    removal_stats["invalid_dna_sequence"] += 1
                    continue

            valid_sequences.append((header, sequence))

        self.logger.info(
            f"After broken entry removal: {len(valid_sequences)} sequences"
        )

        # Parse headers and standardize
        parsed_sequences = []
        species_counts = Counter()
        biotype_counts = Counter()

        for header, sequence in valid_sequences:
            species, biotype, description = parse_fasta_header(header)

            # Check for unknown species/biotypes before standardization
            if self.remove_unknown_species and is_unknown_species(species):
                removal_stats["unknown_species"] += 1
                continue

            if self.remove_unknown_biotypes and is_unknown_biotype(biotype):
                removal_stats["unknown_biotype"] += 1
                continue

            # Standardize names
            species = standardize_species(species, SPECIES_STANDARDIZATION)
            biotype = standardize_biotype(biotype, BIOTYPE_ONTOLOGY)

            # Check again after standardization (in case standardization revealed unknowns)
            if self.remove_unknown_species and is_unknown_species(species):
                removal_stats["unknown_species_post_standardization"] += 1
                continue

            if self.remove_unknown_biotypes and is_unknown_biotype(biotype):
                removal_stats["unknown_biotype_post_standardization"] += 1
                continue

            parsed_sequences.append((species, biotype, description, sequence))
            species_counts[species] += 1
            biotype_counts[biotype] += 1

        self.logger.info(
            f"After unknown entry removal: {len(parsed_sequences)} sequences"
        )
        self.logger.info(
            f"Found {len(species_counts)} species, {len(biotype_counts)} biotypes"
        )

        # Analyze biotype distribution for smart consolidation
        biotype_consolidation = analyze_biotype_distribution(
            biotype_counts, len(parsed_sequences), self.max_biotype_dominance
        )

        # Apply consolidation
        consolidated_sequences = []
        final_species_counts = Counter()
        final_biotype_counts = Counter()

        for species, biotype, description, sequence in parsed_sequences:
            final_biotype = biotype_consolidation.get(biotype, biotype)
            consolidated_sequences.append(
                (species, final_biotype, description, sequence)
            )
            final_species_counts[species] += 1
            final_biotype_counts[final_biotype] += 1

        # Filter out low-quality sequences (high N content)
        quality_filtered = []
        for species, biotype, description, sequence in consolidated_sequences:
            n_content = calculate_n_content(sequence)
            if n_content <= self.max_n_content:
                quality_filtered.append((species, biotype, description, sequence))
            else:
                removal_stats["high_n_content"] += 1

        self.logger.info(
            f"After N-content filtering: {len(quality_filtered)} sequences"
        )

        # Update counts after quality filtering
        final_species_counts = Counter()
        final_biotype_counts = Counter()
        for species, biotype, _, _ in quality_filtered:
            final_species_counts[species] += 1
            final_biotype_counts[biotype] += 1

        # Identify rare categories to remove
        total_after_quality = len(quality_filtered)

        rare_species = filter_rare_categories(
            final_species_counts,
            self.min_species_count,
            self.min_species_proportion,
            total_after_quality,
        )

        rare_biotypes = filter_rare_categories(
            final_biotype_counts,
            self.min_biotype_count,
            self.min_biotype_proportion,
            total_after_quality,
        )

        # Preserve functionally important rare biotypes
        if self.preserve_rare_functional:
            preserved_biotypes = rare_biotypes & self.functional_biotypes
            rare_biotypes = rare_biotypes - self.functional_biotypes
            if preserved_biotypes:
                self.logger.info(
                    f"Preserving rare functional biotypes: {preserved_biotypes}"
                )

        self.logger.info(
            f"Removing {len(rare_species)} rare species, {len(rare_biotypes)} rare biotypes"
        )

        # Final filtering
        final_sequences = []
        for species, biotype, description, sequence in quality_filtered:
            if species in rare_species:
                removal_stats["rare_species"] += 1
            elif biotype in rare_biotypes:
                removal_stats["rare_biotype"] += 1
            else:
                final_sequences.append((species, biotype, description, sequence))

        # Write cleaned dataset
        self._write_sequences(final_sequences, output_file)

        # Generate comprehensive statistics
        final_count = len(final_sequences)
        final_species_counts = Counter(s[0] for s in final_sequences)
        final_biotype_counts = Counter(s[1] for s in final_sequences)

        stats = {
            "original_count": original_count,
            "final_count": final_count,
            "removed_count": original_count - final_count,
            "removal_breakdown": dict(removal_stats),
            "final_species_count": len(final_species_counts),
            "final_biotype_count": len(final_biotype_counts),
            "biotype_consolidations": len(
                [k for k, v in biotype_consolidation.items() if k != v]
            ),
            "top_species": final_species_counts.most_common(10),
            "top_biotypes": final_biotype_counts.most_common(10),
            "removed_species": list(rare_species),
            "removed_biotypes": list(rare_biotypes),
            "species_coverage_final": self._calculate_coverage_stats(
                final_species_counts
            ),
            "biotype_coverage_final": self._calculate_coverage_stats(
                final_biotype_counts
            ),
        }

        self.logger.info(
            f"Enhanced cleaning complete: {original_count} -> {final_count} sequences"
        )
        return stats

    def _calculate_coverage_stats(self, counts: Counter) -> Dict[str, float]:
        """Calculate coverage statistics for categories."""
        total = sum(counts.values())
        if total == 0:
            return {}

        sorted_counts = sorted(counts.values(), reverse=True)
        cumulative = 0
        coverage_stats = {}

        for i, count in enumerate(sorted_counts):
            cumulative += count
            coverage_pct = cumulative / total

            if i + 1 == 1:
                coverage_stats["top_1_coverage"] = coverage_pct
            elif i + 1 == 5:
                coverage_stats["top_5_coverage"] = coverage_pct
            elif i + 1 == 10:
                coverage_stats["top_10_coverage"] = coverage_pct
                break

        return coverage_stats

    def _read_sequences(self, filename: str) -> List[Tuple[str, str]]:
        """Read sequences from FASTA file."""
        sequences = []
        opener = gzip.open if filename.endswith(".gz") else open
        mode = "rt" if filename.endswith(".gz") else "r"

        with opener(filename, mode) as f:
            header = None
            sequence_parts = []

            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line.startswith(">"):
                    if header is not None:
                        sequences.append((header, "".join(sequence_parts)))
                    header = line
                    sequence_parts = []
                elif header is not None:  # Only collect sequence if we have a header
                    sequence_parts.append(line)
                elif line:  # Non-empty line without header
                    self.logger.warning(
                        f"Line {line_num}: sequence data without header, skipping"
                    )

            # Handle last sequence
            if header is not None:
                sequences.append((header, "".join(sequence_parts)))

        return sequences

    def _write_sequences(
        self, sequences: List[Tuple[str, str, str, str]], filename: str
    ):
        """Write cleaned sequences to FASTA file."""
        opener = gzip.open if filename.endswith(".gz") else open
        mode = "wt" if filename.endswith(".gz") else "w"

        with opener(filename, mode) as f:
            for species, biotype, description, sequence in sequences:
                # Format header consistently
                if description and description.strip():
                    header = f">{species}|{biotype}|{description.strip()}"
                else:
                    header = f">{species}|{biotype}|"

                f.write(f"{header}\n{sequence}\n")


def main():
    """Main function with enhanced command line interface."""
    parser = argparse.ArgumentParser(
        description="Smart cleaner for genomic sequence datasets with unknown/broken entry detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input", help="Input FASTA file (can be gzipped)")
    parser.add_argument("output", help="Output cleaned FASTA file")

    # Filtering parameters
    filter_group = parser.add_argument_group("Filtering Parameters")
    filter_group.add_argument(
        "--min-species-count",
        type=int,
        default=10,
        help="Minimum count for species inclusion",
    )
    filter_group.add_argument(
        "--min-biotype-count",
        type=int,
        default=5,
        help="Minimum count for biotype inclusion",
    )
    filter_group.add_argument(
        "--min-species-proportion",
        type=float,
        default=0.001,
        help="Minimum proportion for species inclusion",
    )
    filter_group.add_argument(
        "--min-biotype-proportion",
        type=float,
        default=0.0005,
        help="Minimum proportion for biotype inclusion",
    )
    filter_group.add_argument(
        "--max-n-content", type=float, default=50.0, help="Maximum N content percentage"
    )
    filter_group.add_argument(
        "--max-biotype-dominance",
        type=float,
        default=0.5,
        help="Maximum fraction any biotype can represent",
    )
    filter_group.add_argument(
        "--min-valid-dna-bases",
        type=float,
        default=0.8,
        help="Minimum fraction of valid DNA bases required",
    )

    # Unknown/broken entry detection
    detection_group = parser.add_argument_group("Unknown/Broken Entry Detection")
    detection_group.add_argument(
        "--keep-unknown-species",
        action="store_true",
        help="Keep entries with unknown species names",
    )
    detection_group.add_argument(
        "--keep-unknown-biotypes",
        action="store_true",
        help="Keep entries with unknown biotype names",
    )
    detection_group.add_argument(
        "--keep-broken-sequences",
        action="store_true",
        help="Keep broken/invalid sequences",
    )
    detection_group.add_argument(
        "--keep-broken-headers",
        action="store_true",
        help="Keep entries with broken headers",
    )

    # Behavior options
    behavior_group = parser.add_argument_group("Behavior Options")
    behavior_group.add_argument(
        "--no-preserve-functional",
        action="store_true",
        help="Don't preserve rare functional biotypes",
    )
    behavior_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    behavior_group.add_argument(
        "--report", help="Output detailed statistics report file"
    )
    behavior_group.add_argument(
        "--removal-report", help="Output detailed removal reasons report"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Create cleaner with enhanced parameters
    cleaner = SmartDatasetCleaner(
        min_species_count=args.min_species_count,
        min_biotype_count=args.min_biotype_count,
        min_species_proportion=args.min_species_proportion,
        min_biotype_proportion=args.min_biotype_proportion,
        max_n_content=args.max_n_content,
        max_biotype_dominance=args.max_biotype_dominance,
        preserve_rare_functional=not args.no_preserve_functional,
        remove_unknown_species=not args.keep_unknown_species,
        remove_unknown_biotypes=not args.keep_unknown_biotypes,
        remove_broken_sequences=not args.keep_broken_sequences,
        remove_broken_headers=not args.keep_broken_headers,
        min_valid_dna_bases=args.min_valid_dna_bases,
    )

    # Clean dataset
    try:
        stats = cleaner.clean_dataset(args.input, args.output)

        # Print comprehensive summary
        print("\n" + "=" * 70)
        print("ENHANCED DATASET CLEANING SUMMARY")
        print("=" * 70)
        print(f"Original sequences: {stats['original_count']:,}")
        print(f"Final sequences: {stats['final_count']:,}")
        print(
            f"Removed sequences: {stats['removed_count']:,} ({stats['removed_count']/stats['original_count']*100:.1f}%)"
        )

        print(f"\nDetailed removal breakdown:")
        removal_breakdown = stats["removal_breakdown"]
        total_removed_detailed = sum(removal_breakdown.values())
        for reason, count in sorted(removal_breakdown.items()):
            pct = count / stats["original_count"] * 100
            print(f"  {reason.replace('_', ' ').title()}: {count:,} ({pct:.1f}%)")

        print(f"\nFinal diversity:")
        print(f"  Species: {stats['final_species_count']}")
        print(f"  Biotypes: {stats['final_biotype_count']}")
        print(f"  Biotype consolidations applied: {stats['biotype_consolidations']}")

        # Coverage statistics
        if "species_coverage_final" in stats and stats["species_coverage_final"]:
            print(f"\nSpecies concentration (final dataset):")
            cov = stats["species_coverage_final"]
            if "top_1_coverage" in cov:
                print(f"  Top 1 species: {cov['top_1_coverage']*100:.1f}%")
            if "top_5_coverage" in cov:
                print(f"  Top 5 species: {cov['top_5_coverage']*100:.1f}%")
            if "top_10_coverage" in cov:
                print(f"  Top 10 species: {cov['top_10_coverage']*100:.1f}%")

        if "biotype_coverage_final" in stats and stats["biotype_coverage_final"]:
            print(f"\nBiotype concentration (final dataset):")
            cov = stats["biotype_coverage_final"]
            if "top_1_coverage" in cov:
                print(f"  Top 1 biotype: {cov['top_1_coverage']*100:.1f}%")
            if "top_5_coverage" in cov:
                print(f"  Top 5 biotypes: {cov['top_5_coverage']*100:.1f}%")
            if "top_10_coverage" in cov:
                print(f"  Top 10 biotypes: {cov['top_10_coverage']*100:.1f}%")

        print(f"\nTop species (final dataset):")
        for species, count in stats["top_species"]:
            pct = count / stats["final_count"] * 100
            print(f"  {species}: {count:,} ({pct:.1f}%)")

        print(f"\nTop biotypes (final dataset):")
        for biotype, count in stats["top_biotypes"]:
            pct = count / stats["final_count"] * 100
            print(f"  {biotype}: {count:,} ({pct:.1f}%)")

        # Show examples of removed categories
        if stats["removed_species"]:
            print(f"\nExamples of removed rare species:")
            for species in sorted(stats["removed_species"])[:10]:
                print(f"  {species}")
            if len(stats["removed_species"]) > 10:
                print(f"  ... and {len(stats['removed_species'])-10} more")

        if stats["removed_biotypes"]:
            print(f"\nExamples of removed rare biotypes:")
            for biotype in sorted(stats["removed_biotypes"])[:10]:
                print(f"  {biotype}")
            if len(stats["removed_biotypes"]) > 10:
                print(f"  ... and {len(stats['removed_biotypes'])-10} more")

        # Save detailed reports if requested
        if args.report:
            import json

            with open(args.report, "w") as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"\nDetailed statistics report saved to: {args.report}")

        if args.removal_report:
            with open(args.removal_report, "w") as f:
                f.write("Removal Reason Breakdown\n")
                f.write("=" * 50 + "\n\n")
                for reason, count in sorted(removal_breakdown.items()):
                    pct = count / stats["original_count"] * 100
                    f.write(
                        f"{reason.replace('_', ' ').title()}: {count:,} ({pct:.2f}%)\n"
                    )

                f.write(f"\nRemoved Species ({len(stats['removed_species'])} total):\n")
                f.write("-" * 30 + "\n")
                for species in sorted(stats["removed_species"]):
                    f.write(f"{species}\n")

                f.write(
                    f"\nRemoved Biotypes ({len(stats['removed_biotypes'])} total):\n"
                )
                f.write("-" * 30 + "\n")
                for biotype in sorted(stats["removed_biotypes"]):
                    f.write(f"{biotype}\n")

            print(f"Detailed removal report saved to: {args.removal_report}")

        print(f"\nCleaned dataset saved to: {args.output}")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Error during enhanced cleaning: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
