#!/usr/bin/env python3
import argparse
import os
import sys
import gzip
import re
import random
import logging
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional, Set, Iterator
import time
import math
import glob
import tempfile
import shutil
import subprocess
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import mmap
import bisect

try:
    from pyfaidx import Fasta, FastaIndexingError, FetchError
except ImportError:
    print("Error: pyfaidx library is required. Please install it with 'pip install pyfaidx'", file=sys.stderr)
    sys.exit(1)

# Import plotting libraries
PANDAS_AVAILABLE = False
PLOTTING_AVAILABLE = False
pd = None # Define pd globally

try:
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend for non-interactive environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd # Pandas is essential for many enhanced plots
    PANDAS_AVAILABLE = True
    PLOTTING_AVAILABLE = True # All three (matplotlib, seaborn, pandas) must be available
    sns.set_theme(style="whitegrid", palette="muted") # Apply a pleasant default theme
    logger_init = logging.getLogger(__name__) 
    logger_init.info("Matplotlib, Seaborn, and Pandas are available. Plotting enabled.")
except ImportError:
    logger_init = logging.getLogger(__name__) 
    missing_libs = []
    try: import matplotlib
    except ImportError: missing_libs.append("matplotlib")
    try: import seaborn
    except ImportError: missing_libs.append("seaborn")
    try: import pandas
    except ImportError: missing_libs.append("pandas")

    if missing_libs:
        warning_msg = f"Warning: The following plotting-related libraries are not found: {', '.join(missing_libs)}. Plot generation and some advanced statistics will be skipped or limited."
        print(warning_msg, file=sys.stderr)
        # Logging this warning will be handled in main() after logger is fully configured
    # pd remains None if pandas import failed


# Constants
DNA_ALPHABET = ['A', 'C', 'G', 'T']
VALID_BASES_FOR_STATS = frozenset(DNA_ALPHABET)
BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

EXCLUDED_GFF_TYPES = frozenset({
    "chromosome", "region", "scaffold", "contig", "supercontig",
    "match", "match_part", "CDS_motif", "start_codon", "stop_codon",
    "remark", "sequence_feature", "repeat_region", "STS", "assembly_component", "genome_region", "biological_region"
})

DEFAULT_MIN_LEN = 50
DEFAULT_MAX_LEN = 1000
DEFAULT_TRUNCATION_STRATEGY = "center"
DEFAULT_CONFIDENCE_LEVEL = "medium"
DEFAULT_DESCRIPTION_ATTRIBUTES = "product,Name,description,gene,Note,Dbxref,Ontology_term,inference,function"
PLOT_DPI = 150

CONFIDENCE_LEVELS = {
    "high": {
        "sources": {"RefSeq", "ENSEMBL", "Ensembl", "Havana", "ensembl_havana", "Curated", "TAIR", "SGD", "FlyBase", "WormBase", "MGI", "RGD", "Xenbase", "ZFIN", "VectorBase"},
        "low_confidence_terms": {"prediction", "ab.?initio", "genscan", "glimmer", "augustus", "snap", "geneid", "twinscan", "nscan", "eu.?gene", "fgenesh", "gnomon", "maker", "braker", "evidence.based", "novel.transcript", "automated.annotation", "transcribed_pseudogene", "TEC", "translated_unprocessed_pseudogene", "protein_coding_LoF", "nonsense_mediated_decay"}
    },
    "medium": {
        "sources": {"BestRefSeq", "RefSeq", "ENSEMBL", "Ensembl", "Havana", "ensembl_havana", "WormBase", "FlyBase", "TAIR", "SGD", "RGD", "MGI", "VectorBase", "Xenbase", "ZFIN", "ena", "GenBank", "DDBJ", "EMBL", "Curated", "Community Annotation", "Araport"},
        "low_confidence_terms": {"prediction", "ab.?initio", "genscan", "glimmer", "augustus", "snap", "geneid", "twinscan", "nscan", "eu.?gene", "fgenesh", "gnomon", "maker", "braker", "automated_annotation", "unconfirmed"}
    },
    "low": {"sources": None, "low_confidence_terms": {"very_low_quality_prediction", "highly_fragmented_prediction"}}
}

logger = logging.getLogger(__name__)

# Optimized data structures
@dataclass
class GFFFeature:
    contig: str
    start: int
    end: int
    strand: str
    source: str
    feat_type: str
    attributes: Dict[str, str]
    species: str
    biotype: str
    description: str

@dataclass
class SampleTask:
    species: str
    contig: str
    start: int
    end: int
    strand: str
    category: str
    biotype: str
    description: str
    original_length: int

# Fast sequence operations
_DNA_COMPLEMENT_TABLE = str.maketrans("ACGTNRSYWKMBDHV", "TGCANYSRWMKVHDB")

def reverse_complement(seq: str) -> str:
    return seq.translate(_DNA_COMPLEMENT_TABLE)[::-1]

def calculate_sequence_stats(sequence: str) -> Tuple[Optional[float], Optional[float]]:
    """Optimized calculation of GC content and entropy using lookup tables"""
    if not sequence:
        return None, None
    
    base_counts = Counter(sequence.upper())
    
    valid_count = sum(base_counts[b] for b in VALID_BASES_FOR_STATS)
    if valid_count == 0:
        return None, None
    
    # GC content
    gc_count = base_counts['G'] + base_counts['C']
    gc_content = (gc_count / valid_count) * 100.0
    
    # Shannon entropy
    entropy = 0.0
    for base in VALID_BASES_FOR_STATS:
        count = base_counts[base]
        if count > 0:
            p = count / valid_count
            entropy -= p * math.log2(p)
    
    return gc_content, entropy

def parse_gff_attributes(attr_str: str) -> Dict[str, str]:
    """Parse GFF attributes efficiently"""
    if not attr_str or attr_str == '.':
        return {}
    
    attrs = {}
    for part in attr_str.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            attrs[key.strip()] = value.strip().replace('%2C', ',').replace('%3B', ';').replace('%3D', '=').replace('%25', '%')
    return attrs

def get_species_from_filename(filepath: str) -> str:
    """Extract species name from filename"""
    basename = os.path.basename(filepath)
    normalized = basename.replace('.', ' ').replace('_', ' ')
    
    # Try standard binomial nomenclature
    match = re.search(r"([A-Z][a-z]+) ([a-z]+)", normalized)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    
    # Fallback to simple extraction
    parts = re.sub(r"\.fa.*$|\.gz$", "", basename, flags=re.IGNORECASE)
    parts = parts.replace('.', '_').replace('-', '_')
    alphanum = [p for p in parts.split('_') if p.isalnum() and len(p) > 1]
    
    if alphanum:
        return "_".join(alphanum[:2])[:30]
    return "unknown_species"

class FastGFFParser:
    """Optimized GFF parser with streaming for large files"""
    
    def __init__(self, gff_path: str, trusted_sources: Optional[Set[str]], 
                 low_conf_regex: re.Pattern, desc_attrs: List[str], 
                 target_types: Optional[Set[str]], min_len: int, max_len: int):
        self.gff_path = gff_path
        self.trusted_sources = trusted_sources
        self.low_conf_regex = low_conf_regex
        self.desc_attrs = desc_attrs
        self.target_types = target_types
        self.min_len = min_len
        self.max_len = max_len
        self.is_gzipped = gff_path.endswith('.gz')
        # Pre-compile regex if exists
        self.has_low_conf = bool(self.low_conf_regex.pattern and self.low_conf_regex.pattern != '(?!)')
    
    def parse(self, species: str) -> Tuple[List[GFFFeature], Counter, Set[str]]:
        """Parse GFF file efficiently with streaming"""
        start_time = time.time()
        features = []
        stats = Counter()
        contigs = set()
        lines_processed = 0
        
        logger.info(f"Starting GFF parse for {self.gff_path}")
        
        # Use appropriate file opener
        if self.is_gzipped:
            file_handle = gzip.open(self.gff_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            file_handle = open(self.gff_path, 'r', encoding='utf-8', errors='ignore')
        
        try:
            # Process in chunks for better performance
            for line_num, line in enumerate(file_handle):
                lines_processed += 1
                
                if lines_processed % 100000 == 0:
                    logger.info(f"GFF parser: processed {lines_processed} lines, found {len(features)} valid features")
                
                line = line.strip()
                if not line or line.startswith('#'):
                    stats["comment_or_empty_lines"] += 1
                    continue
                
                fields = line.split('\t')
                if len(fields) != 9:
                    stats["malformed_lines"] += 1
                    continue
                
                contig_raw, source_raw, feat_type_raw, start_str, end_str, _, strand_raw, _, attrs_str_raw = fields
                
                # Collect raw stats before any filtering for specific plots
                stats[f"raw_source_column:{source_raw}"] += 1
                stats[f"raw_feat_type_unfiltered:{feat_type_raw}"] +=1

                # Quick filters before parsing coordinates
                if feat_type_raw.lower() in EXCLUDED_GFF_TYPES:
                    stats["excluded_type"] += 1
                    continue
                
                if self.trusted_sources and source_raw not in self.trusted_sources:
                    stats["untrusted_source"] += 1
                    continue
                
                if self.target_types and feat_type_raw.lower() not in self.target_types:
                    stats["non_target_type"] += 1
                    continue
                
                try:
                    start = int(start_str) - 1 # 0-based start
                    end = int(end_str) # 0-based exclusive end
                except ValueError:
                    stats["invalid_coordinates"] += 1
                    continue
                
                length = end - start
                if length < self.min_len or length > self.max_len or start >= end: # self.max_len is from gff_parser_params
                    stats["length_filtered"] += 1
                    continue
                
                # Low confidence check - only if needed
                if self.has_low_conf:
                    check_str = f"{source_raw}\t{feat_type_raw}\t{attrs_str_raw}".lower()
                    if self.low_conf_regex.search(check_str):
                        stats["low_confidence"] += 1
                        continue
                
                contigs.add(contig_raw)
                stats[f"type:{feat_type_raw}"] += 1 # This is for kept/passed features
                
                # Parse attributes only for kept features 
                attrs = {}
                if attrs_str_raw and attrs_str_raw != '.':
                    needed_attrs = set(self.desc_attrs) | {'biotype', 'gene_biotype', 'transcript_biotype'}
                    for part in attrs_str_raw.split(';'):
                        if '=' in part:
                            k, v = part.split('=', 1)
                            k = k.strip()
                            if k in needed_attrs:
                                attrs[k] = v.strip().replace('%2C', ',').replace('%3B', ';').replace('%3D', '=').replace('%25', '%')
                
                # Extract biotype and description
                biotype = (attrs.get('biotype') or attrs.get('gene_biotype') or 
                          attrs.get('transcript_biotype') or feat_type_raw).replace(' ', '_')
                
                desc_parts = []
                for attr_key in self.desc_attrs: # Use attr_key to avoid conflict
                    if attr_key in attrs and attrs[attr_key] != '.':
                        desc_parts.append(attrs[attr_key])
                description = "; ".join(desc_parts) if desc_parts else ""
                
                features.append(GFFFeature(
                    contig=contig_raw, start=start, end=end, strand=strand_raw,
                    source=source_raw, feat_type=feat_type_raw, attributes=attrs,
                    species=species, biotype=biotype, description=description
                ))
                    
        finally:
            file_handle.close()
        
        stats["total_features_kept"] = len(features)
        stats["total_lines_processed"] = lines_processed
        elapsed = time.time() - start_time
        logger.info(f"GFF parse complete for {self.gff_path}: {lines_processed} lines, {len(features)} features kept in {elapsed:.2f}s")
        
        return features, stats, contigs

class FastaSequenceSampler:
    """Efficient FASTA sequence sampling with batch processing"""
    
    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self.fasta = None
        self._sequences = {}  # Store all sequences in memory
        self._use_memory = True  # Flag to use in-memory sequences
        self._load_sequences()
    
    def _load_sequences(self):
        """Load all sequences into memory for fast access"""
        start_time = time.time()
        logger.info(f"Loading FASTA file into memory: {self.fasta_path}")
        
        total_size = 0
        
        try:
            # Open file
            if self.fasta_path.endswith('.gz'):
                file_handle = gzip.open(self.fasta_path, 'rt')
            else:
                file_handle = open(self.fasta_path, 'r')
            
            current_contig = None
            current_seq = []
            
            with file_handle:
                for line in file_handle:
                    line = line.strip()
                    if line.startswith('>'):
                        # Save previous sequence
                        if current_contig:
                            seq = ''.join(current_seq).upper()
                            self._sequences[current_contig] = seq
                            total_size += len(seq)
                            # logger.debug(f"Loaded {current_contig}: {len(seq)} bp") # Too verbose
                        
                        # Start new sequence
                        current_contig = line[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line)
                
                # Save last sequence
                if current_contig:
                    seq = ''.join(current_seq).upper()
                    self._sequences[current_contig] = seq
                    total_size += len(seq)
            
            elapsed = time.time() - start_time
            # logger.info(f"Loaded {len(self._sequences)} contigs, {total_size/1e9:.2f} GB in {elapsed:.2f}s ({total_size/elapsed/1e6 if elapsed > 0 else 0:.0f} MB/s)")
            logger.info(f"Loaded {len(self._sequences)} contigs ({total_size/1e6:.2f} MB total) into memory for {self.fasta_path} in {elapsed:.2f}s.")

            # Check memory usage
            if total_size > 5e9:  # If > 5GB, consider using pyfaidx instead (though we are already in memory mode)
                logger.warning(f"Large genome ({total_size/1e9:.2f} GB) loaded into memory for {self.fasta_path}. This might be slow or lead to OOM issues for very large files.")
            
        except Exception as e:
            logger.error(f"Failed to load FASTA {self.fasta_path} into memory: {e}")
            logger.info("Falling back to pyfaidx for FASTA operations on this file.")
            self._use_memory = False
            self._sequences.clear() # Ensure memory is freed
            
            # Fall back to pyfaidx
            try:
                # Use read_ahead for potentially better performance with pyfaidx
                self.fasta = Fasta(self.fasta_path, sequence_always_upper=True, 
                                  read_ahead=50*1024*1024) # 50MB read_ahead buffer
            except Exception as e2:
                logger.error(f"Failed to open {self.fasta_path} with pyfaidx after memory load failed: {e2}")
                raise # Re-raise if pyfaidx also fails
    
    def get_contigs(self) -> Set[str]:
        """Get all contig names"""
        if self._use_memory:
            return set(self._sequences.keys())
        else:
            return set(self.fasta.keys()) if self.fasta else set()
    
    def batch_sample_sequences(self, tasks: List[SampleTask], min_len: int, max_len: int,
                              truncation: str = 'center') -> List[Tuple[str, str, float, float, int]]:
        """Batch process sequence sampling for efficiency"""
        start_time = time.time()
        results = []
        total_tasks = len(tasks)
        # logger.info(f"Starting batch sampling of {total_tasks} sequences (memory mode: {self._use_memory}) for {self.fasta_path}")
        
        processed_count = 0
        failed_count = 0 
        
        for task in tasks:
            processed_count += 1
            
            if processed_count % 1000 == 0 and total_tasks > 1000 : # Avoid spamming logs for small batches
                elapsed_time_batch = time.time() - start_time # Renamed
                rate = processed_count / elapsed_time_batch if elapsed_time_batch > 0 else 0
                logger.info(f"Sampling progress for current FASTA: {processed_count}/{total_tasks} ({processed_count/total_tasks*100:.1f}%) at {rate:.0f} seqs/sec, {failed_count} failed")
            
            if task.contig == "random": # This should not happen if random samples are handled separately
                seq = ''.join(np.random.choice(list(DNA_ALPHABET), task.original_length))
                gc, entropy = calculate_sequence_stats(seq)
                header = f">random|random_dna|{task.description}"
                results.append((header, seq, gc, entropy, len(seq)))
            else:
                try:
                    seq = "" # Initialize seq
                    if self._use_memory:
                        if task.contig in self._sequences:
                            # Slicing Python strings is efficient
                            seq = self._sequences[task.contig][task.start:task.end]
                        else:
                            logger.debug(f"Contig {task.contig} not found in in-memory store for {self.fasta_path}")
                            failed_count += 1
                            continue
                    else: # Using pyfaidx
                        if self.fasta and task.contig in self.fasta:
                            seq = self.fasta[task.contig][task.start:task.end].seq
                        else:
                            logger.debug(f"Contig {task.contig} not found via pyfaidx for {self.fasta_path}")
                            failed_count += 1
                            continue
                    
                    if task.strand == '-':
                        seq = reverse_complement(seq)
                    
                    # Apply truncation (max_len here is the overall script's max_len argument)
                    seq_len = len(seq)
                    if seq_len > max_len:
                        if truncation == 'start':
                            seq = seq[:max_len]
                        elif truncation == 'end':
                            seq = seq[-max_len:]
                        elif truncation == 'center':
                            offset = (seq_len - max_len) // 2
                            seq = seq[offset:offset + max_len]
                        elif truncation == 'random_segment':
                            start_pos = random.randint(0, seq_len - max_len)
                            seq = seq[start_pos:start_pos + max_len]
                        # If truncation is 'none', no change here, length check below handles it.
                    
                    final_seq_len = len(seq)
                    if final_seq_len < min_len or (truncation == "none" and final_seq_len > max_len):
                        failed_count += 1
                        continue
                    
                    gc, entropy = calculate_sequence_stats(seq)
                    # Ensure header doesn't get too long
                    base_header = f">{task.species}|{task.biotype}|{task.description}"
                    header = (base_header[:247] + '...') if len(base_header) > 250 else base_header
                    results.append((header, seq, gc, entropy, final_seq_len)) # Use final_seq_len
                    
                except FetchError as fe: # Specific error for pyfaidx
                    logger.debug(f"FetchError for {task.contig}:{task.start}-{task.end} from {self.fasta_path}: {fe}")
                    failed_count += 1
                except KeyError as ke: # Specific error for in-memory dict
                    logger.debug(f"KeyError for {task.contig} from in-memory sequences for {self.fasta_path}: {ke}")
                    failed_count += 1
                except Exception as e: # Catch-all for other errors
                    logger.debug(f"Failed to fetch/process {task.contig}:{task.start}-{task.end} from {self.fasta_path}: {e}")
                    failed_count += 1
        
        elapsed_time_total_batch = time.time() - start_time # Renamed
        # logger.info(f"Batch sampling for {self.fasta_path} complete: {len(results)}/{total_tasks} sequences in {elapsed_time_total_batch:.2f}s ({len(results)/elapsed_time_total_batch if elapsed_time_total_batch > 0 else 0:.0f} seqs/sec), {failed_count} failed")
        
        return results
    
    def close(self):
        """Close FASTA file and clear memory"""
        self._sequences.clear() # Explicitly clear the dictionary
        if self.fasta:
            try:
                self.fasta.close()
            except: # pyfaidx close can sometimes error if already closed or in a weird state
                pass

def process_species_files(species_data: Dict[str, Dict[str, Any]], 
                         gff_parser_params: Dict[str, Any],
                         num_annotated: int, num_random: int,
                         min_len: int, max_len: int, 
                         truncation: str) -> Dict[str, Any]:
    """Process all species files and generate samples with optimized selection"""
    
    all_sample_tasks_from_gff = [] 
    all_gff_stats_aggregated = Counter() 
    all_contig_mismatches_per_species = {} 
    species_processing_summary = {} 
    species_coverage_stats = {} # For the annotation coverage plot

    # First pass: validate FASTAs and parse GFFs for each species
    for species, file_info in species_data.items():
        fasta_path = file_info['fasta']
        gff_paths = file_info.get('gffs', [])
        
        current_species_fasta_contigs = set()
        try:
            # Temporarily open FASTA to get contig names, FastaSequenceSampler will handle full load later if needed
            # This sampler instance is just for contig name validation here.
            temp_sampler_for_contigs = FastaSequenceSampler(fasta_path)
            current_species_fasta_contigs = temp_sampler_for_contigs.get_contigs()
            temp_sampler_for_contigs.close() # Close it immediately
        except Exception as e:
            logger.error(f"Failed to read FASTA {fasta_path} for species {species}: {e}")
            species_processing_summary[species] = {"candidates_initial": 0, "candidates_valid": 0, "samples_selected_annotated": 0, "error": str(e)}
            continue # Skip this species if FASTA is unreadable
        
        if not current_species_fasta_contigs:
            logger.error(f"No contigs found in FASTA: {fasta_path} for species {species}")
            species_processing_summary[species] = {"candidates_initial": 0, "candidates_valid": 0, "samples_selected_annotated": 0, "error": "No contigs in FASTA"}
            continue
        
        species_gff_features_unfiltered = [] # All features from GFF files for this species
        species_gff_contigs_from_all_gffs = set() # All contig names mentioned in GFFs for this species
        
        for gff_path in gff_paths:
            if not os.path.exists(gff_path):
                logger.warning(f"GFF file not found, skipping: {gff_path} for species {species}")
                continue
            
            try:
                # Pass the overall script's min_len and max_len to the parser
                # If truncation_strategy is 'none', parser max_len is script's max_len.
                # Otherwise, parser max_len is effectively infinite, and sampler truncates.
                parser_max_len = max_len if truncation == "none" else float('inf')
                current_gff_parser_params = {**gff_parser_params, 'min_len': min_len, 'max_len': parser_max_len}

                parser = FastGFFParser(gff_path, **current_gff_parser_params)
                features_from_parser, single_gff_stats, contigs_from_single_gff = parser.parse(species)
                
                species_gff_features_unfiltered.extend(features_from_parser)
                species_gff_contigs_from_all_gffs.update(contigs_from_single_gff)
                all_gff_stats_aggregated.update(single_gff_stats)
                
                logger.info(f"Parsed {len(features_from_parser)} features from {gff_path} for species {species}.")
            except Exception as e:
                logger.error(f"Failed to parse GFF {gff_path} for species {species}: {e}")
        
        valid_gff_features_on_fasta_contigs = []
        contigs_with_valid_gff_features = set()

        for feature in species_gff_features_unfiltered:
            if feature.contig in current_species_fasta_contigs:
                # Now, apply the true min/max_len filtering considering truncation strategy
                # The parser might have passed features > max_len if truncation is not 'none'
                # The sampler will handle truncation. Here we ensure it *could* be sampled.
                original_feat_len = feature.end - feature.start
                passes_final_length_check = False
                if truncation == "none":
                    if min_len <= original_feat_len <= max_len:
                        passes_final_length_check = True
                else: # Truncation will happen, so if original_len >= min_len, it's a candidate
                    if original_feat_len >= min_len:
                        passes_final_length_check = True
                
                if passes_final_length_check:
                    valid_gff_features_on_fasta_contigs.append(feature)
                    contigs_with_valid_gff_features.add(feature.contig)
                    all_sample_tasks_from_gff.append(SampleTask(
                        species=species,
                        contig=feature.contig,
                        start=feature.start,
                        end=feature.end,
                        strand=feature.strand,
                        category="annotated", 
                        biotype=feature.biotype,
                        description=feature.description,
                        original_length=original_feat_len
                    ))
        
        all_contig_mismatches_per_species[species] = {
            "gff_total": len(species_gff_contigs_from_all_gffs),
            "fasta_total": len(current_species_fasta_contigs),
            "gff_in_fasta": len(species_gff_contigs_from_all_gffs & current_species_fasta_contigs),
            "gff_not_in_fasta": len(species_gff_contigs_from_all_gffs - current_species_fasta_contigs),
            "fasta_not_in_gff": len(current_species_fasta_contigs - species_gff_contigs_from_all_gffs)
        }
        
        species_coverage_stats[species] = (len(contigs_with_valid_gff_features) / len(current_species_fasta_contigs)) * 100.0 if current_species_fasta_contigs else 0.0
        
        species_processing_summary[species] = {
            "candidates_initial": len(species_gff_features_unfiltered), 
            "candidates_valid": len(valid_gff_features_on_fasta_contigs), 
            "samples_selected_annotated": 0 
        }
    
    selected_annotated_samples = []
    
    if all_sample_tasks_from_gff and num_annotated > 0:
        species_counts_in_tasks = Counter(task.species for task in all_sample_tasks_from_gff)
        
        if len(all_sample_tasks_from_gff) <= num_annotated:
            selected_annotated_samples = all_sample_tasks_from_gff
            for task in selected_annotated_samples: 
                if task.species in species_processing_summary:
                    species_processing_summary[task.species]["samples_selected_annotated"] += 1
        else:
            total_tasks_available = len(all_sample_tasks_from_gff)
            proportions = {sp: count / total_tasks_available for sp, count in species_counts_in_tasks.items()}
            allocations = {sp: math.floor(prop * num_annotated) for sp, prop in proportions.items()}
            
            remainder = num_annotated - sum(allocations.values())
            if remainder > 0:
                fractional_parts = {sp: (prop * num_annotated) - allocations[sp] for sp, prop in proportions.items()}
                sorted_species_by_fraction = sorted(fractional_parts.keys(), key=lambda sp_key: fractional_parts[sp_key], reverse=True)
                for i in range(remainder):
                    allocations[sorted_species_by_fraction[i % len(sorted_species_by_fraction)]] += 1 
            
            tasks_grouped_by_species = defaultdict(list)
            for task in all_sample_tasks_from_gff:
                tasks_grouped_by_species[task.species].append(task)
            
            for species_key, allocated_count in allocations.items():
                features_for_this_species = tasks_grouped_by_species[species_key]
                num_to_select_this_species = min(allocated_count, len(features_for_this_species))
                if num_to_select_this_species > 0:
                    selected_batch = random.sample(features_for_this_species, num_to_select_this_species)
                    selected_annotated_samples.extend(selected_batch)
                    if species_key in species_processing_summary: 
                         species_processing_summary[species_key]["samples_selected_annotated"] = num_to_select_this_species
    
    random_samples_tasks = []
    for _ in range(num_random):
        length = random.randint(min_len, max_len)
        random_samples_tasks.append(SampleTask(
            species="random", 
            contig="random",  
            start=0,
            end=length,
            strand="+",
            category="random",
            biotype="random_dna",
            description=f"Randomly generated DNA L{length}",
            original_length=length
        ))
    
    return {
        "annotated_samples": selected_annotated_samples,
        "random_samples": random_samples_tasks,
        "gff_stats": all_gff_stats_aggregated,
        "contig_mismatches": all_contig_mismatches_per_species,
        "species_stats": species_processing_summary,
        "species_coverage_stats": species_coverage_stats
    }

def write_samples(output_path: str, species_data: Dict[str, Dict[str, Any]],
                  annotated_samples: List[SampleTask], random_samples: List[SampleTask],
                  min_len: int, max_len: int, truncation: str) -> Dict[str, Any]:
    """Write samples to output FASTA using batch processing"""
    
    write_start_time = time.time() # Renamed
    logger.info(f"=== Writing samples to {output_path} ===")
    
    # Open output file
    if output_path.endswith('.gz'):
        out_handle = gzip.open(output_path, 'wt')
    else:
        out_handle = open(output_path, 'wt')
    
    # Collect statistics for reporting
    final_metadata_for_report = [] # Renamed
    overall_sampling_stats = Counter() # Renamed
    annotated_lengths_original = [] # Renamed
    annotated_lengths_final_sampled = [] # Renamed
    
    try:
        # Group annotated samples by species for efficient FASTA file handling
        annotated_samples_by_species = defaultdict(list)
        for sample_task in annotated_samples: # Renamed var
            annotated_samples_by_species[sample_task.species].append(sample_task)
        
        total_written_count = 0 # Renamed
        
        # Process annotated samples species by species
        logger.info(f"Processing {len(annotated_samples_by_species)} species groups for annotated samples.")
        for species, species_sample_tasks in annotated_samples_by_species.items(): # Renamed vars
            species_batch_start_time = time.time() # Renamed
            
            fasta_path_for_species = species_data.get(species, {}).get('fasta') # Renamed
            if not fasta_path_for_species:
                logger.error(f"No FASTA path found for species {species} during writing. Skipping {len(species_sample_tasks)} samples.")
                overall_sampling_stats[f"annotated_skipped_no_fasta_at_write:{species}"] += len(species_sample_tasks)
                continue
                
            logger.info(f"Writing {len(species_sample_tasks)} annotated samples for species {species} from {fasta_path_for_species}")
            
            sequence_sampler_for_species = FastaSequenceSampler(fasta_path_for_species) # Renamed
            
            batch_results_for_species = sequence_sampler_for_species.batch_sample_sequences(
                species_sample_tasks, min_len, max_len, truncation
            )
            
            for i, (header, seq, gc, entropy, sampled_len) in enumerate(batch_results_for_species):
                original_task = species_sample_tasks[i] 
                
                out_handle.write(f"{header}\n{seq}\n")
                final_metadata_for_report.append((header, "annotated", gc, entropy, sampled_len, original_task.original_length, original_task.species, original_task.biotype))
                
                overall_sampling_stats[f"annotated_species:{species}"] += 1
                overall_sampling_stats[f"annotated_biotype:{original_task.biotype}"] += 1
                annotated_lengths_original.append(original_task.original_length)
                annotated_lengths_final_sampled.append(sampled_len)
                total_written_count += 1
            
            sequence_sampler_for_species.close() # Close FASTA file for this species
            logger.info(f"Species {species} (annotated) written in {time.time() - species_batch_start_time:.2f}s")

        if random_samples:
            logger.info(f"Writing {len(random_samples)} random samples")
            random_batch_start_time = time.time()
            for random_task in random_samples: # Renamed var
                seq_random = "".join(np.random.choice(list(DNA_ALPHABET), random_task.original_length))
                gc_random, entropy_random = calculate_sequence_stats(seq_random)
                
                header_random = f">{random_task.species}|{random_task.biotype}|{random_task.description}"
                header_random = (header_random[:247] + '...') if len(header_random) > 250 else header_random

                out_handle.write(f"{header_random}\n{seq_random}\n")
                final_metadata_for_report.append((header_random, "random", gc_random, entropy_random, len(seq_random), random_task.original_length, random_task.species, random_task.biotype))
                
                overall_sampling_stats["random_count"] += 1
                total_written_count += 1
            logger.info(f"Random samples written in {time.time() - random_batch_start_time:.2f}s")
        
        logger.info(f"Total samples written to {output_path}: {total_written_count} in {time.time() - write_start_time:.2f}s")
    
    finally:
        out_handle.close()
    
    overall_sampling_stats["total_written"] = total_written_count
    
    return {
        "metadata": final_metadata_for_report,
        "sampling_stats": overall_sampling_stats,
        "annotated_lengths_orig": annotated_lengths_original,
        "annotated_lengths_sampled": annotated_lengths_final_sampled
    }

def plot_length_distribution(lengths: List[int], title: str, filepath: str, bins: int = 50):
    if not PLOTTING_AVAILABLE or not lengths: 
        logger.debug(f"Skipping plot '{title}': plotting unavailable or no data.")
        return False
    plt.figure(figsize=(10, 6)); sns.histplot(lengths, bins=bins, kde=True)
    plt.title(title); plt.xlabel("Sequence Length (bp)"); plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    try: plt.savefig(filepath, dpi=PLOT_DPI); plt.close(); logger.info(f"Plot saved: {filepath}"); return True
    except Exception as e: logger.error(f"Failed to save plot {filepath}: {e}"); plt.close(); return False

def plot_categorical_distribution(counts: Counter, title: str, xlabel: str, filepath: str, top_n: int = 20):
    if not PLOTTING_AVAILABLE or not counts: 
        logger.debug(f"Skipping plot '{title}': plotting unavailable or no data.")
        return False
    if not isinstance(counts, Counter): counts = Counter(counts)
    if not counts: 
        logger.debug(f"Skipping plot '{title}': no data in Counter.")
        return False
    common_items = counts.most_common(top_n)
    if not common_items: 
        logger.debug(f"Skipping plot '{title}': no items after most_common filter.")
        return False
    labels, values = zip(*common_items)
    fig_width = max(12, len(labels) * 0.7 if len(labels) > 0 else 12)
    plt.figure(figsize=(fig_width, 7))
    # Address Seaborn FutureWarning for palette without hue
    barplot = sns.barplot(x=list(labels), y=list(values), hue=list(labels) if labels else None, palette="viridis", legend=False)
    plt.title(f"{title} (Top {min(top_n, len(labels))})"); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right", fontsize=min(10, 90/len(labels) if len(labels)>10 else 9))
    for i, v_val in enumerate(values): barplot.text(i, v_val + 0.5, str(v_val), color='black', ha="center", va="bottom", fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    try: plt.savefig(filepath, dpi=PLOT_DPI); plt.close(); logger.info(f"Plot saved: {filepath}"); return True
    except Exception as e: logger.error(f"Failed to save plot {filepath}: {e}"); plt.close(); return False

def plot_scatter_lengths(original_lengths: List[int], sampled_lengths: List[int], title: str, filepath: str):
    if not PLOTTING_AVAILABLE or not original_lengths or not sampled_lengths or len(original_lengths) != len(sampled_lengths):
        logger.debug(f"Skipping scatter plot '{title}': plotting libraries unavailable or insufficient/mismatched data.")
        return False
    plt.figure(figsize=(10, 8))
    plt.scatter(original_lengths, sampled_lengths, alpha=0.3, s=15, edgecolors='w', linewidths=0.3)
    min_val_orig = min(original_lengths, default=0)
    max_val_orig = max(original_lengths, default=1)
    min_val_samp = min(sampled_lengths, default=0)
    max_val_samp = max(sampled_lengths, default=1)
    
    min_val = min(min_val_orig, min_val_samp)
    max_val = max(max_val_orig, max_val_samp)

    if min_val < max_val : plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label="y=x (No length change)")
    plt.title(title); plt.xlabel("Original Feature Length (bp)"); plt.ylabel("Final Sampled Length (bp)")
    plt.xscale("log"); plt.yscale("log")
    plt.grid(True, linestyle=':', alpha=0.7); plt.legend(); plt.tight_layout()
    try: plt.savefig(filepath, dpi=PLOT_DPI); plt.close(); logger.info(f"Scatter plot saved: {filepath}"); return True
    except Exception as e: logger.error(f"Failed to save scatter plot {filepath}: {e}"); plt.close(); return False

def plot_numerical_dist_by_category(data_dict: Dict[str, List[float]], value_name: str, category_name: str, title_override: Optional[str], filepath: str, top_n_categories: int = 15, plot_type: str = "violin"):
    if not PLOTTING_AVAILABLE or not PANDAS_AVAILABLE or not data_dict:
        logger.debug(f"Skipping '{value_name}' by '{category_name}' plot: plotting/pandas unavailable or no data.")
        return False
    plot_data_list = []
    for cat_val, val_list in data_dict.items():
        if val_list:
            for val_item in val_list: 
                 if val_item is not None and not math.isnan(val_item): # Ensure data is valid
                    plot_data_list.append({category_name: cat_val, value_name: val_item})
    if not plot_data_list: 
        logger.debug(f"Skipping '{value_name}' by '{category_name}' plot: no valid data after processing."); return False
    
    df = pd.DataFrame(plot_data_list)
    if df.empty or df[value_name].isnull().all():
         logger.debug(f"Skipping '{value_name}' by '{category_name}' plot: DataFrame empty or all values NaN."); return False

    category_counts = df[category_name].value_counts()
    actual_top_n = min(top_n_categories, len(category_counts))
    if actual_top_n == 0: 
        logger.debug(f"Skipping '{value_name}' by '{category_name}' plot: no categories with data."); return False
    
    top_categories_list = category_counts.nlargest(actual_top_n).index.tolist()
    df_filtered = df[df[category_name].isin(top_categories_list)]
    if df_filtered.empty or df_filtered[value_name].isnull().all(): 
        logger.debug(f"Skipping '{value_name}' by '{category_name}' plot: no data after filtering for top categories or all values NaN."); return False
    
    num_unique_cats_to_plot = len(df_filtered[category_name].unique())
    fig_width = max(12, num_unique_cats_to_plot * 0.9)
    plt.figure(figsize=(fig_width, 7))
    ordered_categories_list = category_counts.nlargest(actual_top_n).index
    plot_func = sns.violinplot if plot_type == "violin" else sns.boxplot
    
    # Address Seaborn FutureWarning for palette/scale
    plot_kwargs = {"x": category_name, "y": value_name, "data": df_filtered, 
                   "hue": category_name, "palette": "muted", 
                   "order": ordered_categories_list, "legend": False}
    if plot_type == "violin": 
        plot_kwargs.update({"cut":0, "inner":"quartile", "density_norm":"width"}) # "scale" changed to "density_norm"
    
    try:
        plot_func(**plot_kwargs)
    except Exception as e_plot: # Catch issues within seaborn/matplotlib
        logger.error(f"Error during actual plotting for '{title_override or value_name}': {e_plot}"); plt.close(); return False

    plot_title = title_override if title_override else f"Distribution of {value_name} by {category_name} (Top {num_unique_cats_to_plot})"
    plt.title(plot_title); plt.xlabel(category_name); plt.ylabel(value_name)
    plt.xticks(rotation=45, ha="right", fontsize=min(10, 90/num_unique_cats_to_plot if num_unique_cats_to_plot > 10 else 9))
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    try: plt.savefig(filepath, dpi=PLOT_DPI); plt.close(); logger.info(f"Plot saved: {filepath}"); return True
    except Exception as e: logger.error(f"Failed to save plot {filepath}: {e}"); plt.close(); return False

def plot_2d_density(x_data: List[float], y_data: List[float], x_label: str, y_label: str, title: str, filepath: str, color_data: Optional[List[str]] = None):
    if not PLOTTING_AVAILABLE or not PANDAS_AVAILABLE or not x_data or not y_data or len(x_data) != len(y_data):
        logger.debug(f"Skipping 2D density plot '{title}': plotting/pandas unavailable or insufficient/mismatched data.")
        return False
    plt.figure(figsize=(10, 8))
    valid_indices = [i for i, (x, y) in enumerate(zip(x_data, y_data)) if x is not None and y is not None and not (math.isnan(x) or math.isnan(y))]
    if not valid_indices: 
        logger.debug(f"Skipping 2D density plot '{title}': no valid data points after NaN/None filtering."); return False
    
    x_data_clean = [x_data[i] for i in valid_indices]
    y_data_clean = [y_data[i] for i in valid_indices]
    
    if not x_data_clean or not y_data_clean : 
        logger.debug(f"Skipping 2D density plot '{title}': cleaned data lists are empty."); return False
        
    color_data_clean = None
    if color_data:
        if len(color_data) == len(x_data): # Original length for color_data
            color_data_clean = [color_data[i] for i in valid_indices]
            if not any(color_data_clean): color_data_clean = None # If all are None/empty after filtering
        else:
             logger.warning(f"Length mismatch between data and color_data for plot '{title}'. Skipping color.")


    if color_data_clean:
        unique_colors = sorted(list(set(c for c in color_data_clean if c))) # Filter out None for unique colors
        if not unique_colors : # Fallback if all color data was None
            color_data_clean = None 
        else:
            palette = sns.color_palette("viridis", n_colors=max(1,len(unique_colors)))
            df_plot = pd.DataFrame({x_label: x_data_clean, y_label: y_data_clean, 'Category': color_data_clean})
            if df_plot.empty: logger.debug(f"DataFrame for scatter plot '{title}' is empty."); return False
            sns.scatterplot(x=x_label, y=y_label, hue='Category', data=df_plot, palette=palette, s=20, alpha=0.5, legend="full")
    
    if not color_data_clean: # If no color data or it was unsuitable
        try:
            if len(set(x_data_clean)) <= 1 or len(set(y_data_clean)) <= 1:
                 logger.warning(f"Not enough variance in cleaned data for hexbin plot '{title}'. Falling back to scatter.")
                 plt.scatter(x_data_clean, y_data_clean, alpha=0.3, s=10)
            else:
                 hexbin_bins_arg: Optional[str] = 'log' if len(x_data_clean)>1000 else None
                 hb = plt.hexbin(x_data_clean, y_data_clean, gridsize=50, cmap='Blues', mincnt=1, bins=hexbin_bins_arg) # type: ignore
                 label_colorbar = 'Count in bin'
                 if hb.get_array().min() > 0 and hexbin_bins_arg == 'log':
                     label_colorbar = 'log10(Count in bin)'
                 plt.colorbar(hb, label=label_colorbar)
        except Exception as e_hex:
            logger.warning(f"Hexbin plot failed for '{title}': {e_hex}. Falling back to scatter.")
            plt.scatter(x_data_clean, y_data_clean, alpha=0.3, s=10)

    plt.title(title); plt.xlabel(x_label); plt.ylabel(y_label)
    plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
    try: plt.savefig(filepath, dpi=PLOT_DPI); plt.close(); logger.info(f"Plot saved: {filepath}"); return True
    except Exception as e: logger.error(f"Failed to save 2D density plot {filepath}: {e}"); plt.close(); return False

def plot_length_distribution_comparison(length_data: Dict[str, List[int]], title: str, filepath: str):
    if not PLOTTING_AVAILABLE or not PANDAS_AVAILABLE or not length_data:
        logger.debug(f"Skipping length distribution comparison plot '{title}': plotting/pandas unavailable or no data.")
        return False
    plt.figure(figsize=(12, 7))
    df_parts = []
    for category, lengths in length_data.items():
        if lengths and category != "overall": # Exclude "overall" if present, as it would dominate
            df_parts.append(pd.DataFrame({'Length': lengths, 'Category': category}))
    if not df_parts: 
        logger.debug(f"Skipping '{title}': No data in categories for comparison plot."); return False
    df_plot = pd.concat(df_parts)
    if df_plot.empty:
        logger.debug(f"Skipping '{title}': DataFrame empty after concat for comparison plot."); return False
    sns.kdeplot(data=df_plot, x='Length', hue='Category', fill=True, alpha=0.5, common_norm=False, warn_singular=False)
    plt.title(title); plt.xlabel("Sequence Length (bp)"); plt.ylabel("Density")
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    try: plt.savefig(filepath, dpi=PLOT_DPI); plt.close(); logger.info(f"Plot saved: {filepath}"); return True
    except Exception as e: logger.error(f"Failed to save plot {filepath}: {e}"); plt.close(); return False

def plot_annotation_coverage_per_species(coverage_data: Dict[str, float], filepath: str, top_n: int = 25):
    if not PLOTTING_AVAILABLE or not PANDAS_AVAILABLE or not coverage_data:
        logger.debug("Skipping annotation coverage plot: plotting/pandas unavailable or no data.")
        return False
    # coverage_data is {species: percentage_coverage}
    sorted_species = sorted(coverage_data.items(), key=lambda item: item[1], reverse=True)[:top_n]
    if not sorted_species: 
        logger.debug("Skipping annotation coverage plot: no data after sorting/top_n."); return False
    
    species_names, coverages = zip(*sorted_species)

    fig_width = max(10, len(species_names) * 0.5)
    plt.figure(figsize=(fig_width, 7))
    # Address Seaborn FutureWarning for palette without hue
    barplot = sns.barplot(x=list(species_names), y=list(coverages), hue=list(species_names) if species_names else None, palette="coolwarm_r", legend=False)
    plt.title(f"Proportion of FASTA Contigs with GFF Features (Top {len(species_names)})")
    plt.xlabel("Species"); plt.ylabel("Proportion of FASTA Contigs with GFF Features (%)")
    plt.xticks(rotation=45, ha="right", fontsize=min(10, 90/len(species_names) if len(species_names)>10 else 9))
    plt.ylim(0, max(100.0, max(coverages) * 1.1 if coverages else 100.0))
    for i, v_val in enumerate(coverages): barplot.text(i, v_val + 0.5, f"{v_val:.1f}%", color='black', ha="center", va="bottom", fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    try: plt.savefig(filepath, dpi=PLOT_DPI); plt.close(); logger.info(f"Plot saved: {filepath}"); return True
    except Exception as e: logger.error(f"Failed to save plot {filepath}: {e}"); plt.close(); return False

def plot_contig_mismatch_summary(mismatch_stats: Dict[str, Dict[str, int]], filepath: str, top_n: int = 25):
    if not PLOTTING_AVAILABLE or not PANDAS_AVAILABLE or not mismatch_stats:
        logger.debug("Skipping contig mismatch plot: plotting/pandas unavailable or no data.")
        return False
    sorted_species_keys = sorted(mismatch_stats.keys(), key=lambda sp: mismatch_stats[sp].get('gff_total', 0) + mismatch_stats[sp].get('fasta_total', 0), reverse=True)[:top_n]
    if not sorted_species_keys: 
        logger.debug("Skipping contig mismatch plot: No species data."); return False
    plot_data = []
    for sp_key in sorted_species_keys:
        stats = mismatch_stats[sp_key]
        plot_data.append({'Species': sp_key, 'Count': stats.get('gff_in_fasta', 0), 'Category': 'GFF Contigs in FASTA (Common)'})
        plot_data.append({'Species': sp_key, 'Count': stats.get('gff_not_in_fasta', 0), 'Category': 'GFF Contigs Only'})
        plot_data.append({'Species': sp_key, 'Count': stats.get('fasta_not_in_gff', 0), 'Category': 'FASTA Contigs Only'})
    df = pd.DataFrame(plot_data)
    if df.empty: 
        logger.debug("Skipping contig mismatch plot: DataFrame empty."); return False
    fig_width = max(12, len(sorted_species_keys) * 0.7)
    plt.figure(figsize=(fig_width, 8))
    sns.barplot(x='Species', y='Count', hue='Category', data=df, dodge=True, palette="Set2")
    plt.title(f"GFF/FASTA Contig Name Concordance (Top {len(sorted_species_keys)} Species by Total Contigs)")
    plt.xlabel("Species"); plt.ylabel("Number of Contigs")
    plt.xticks(rotation=60, ha="right", fontsize=min(10, 80/len(sorted_species_keys) if len(sorted_species_keys)>10 else 9))
    plt.legend(title='Contig Source/Status', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout(rect=[0, 0, 0.85, 1])
    try: plt.savefig(filepath, dpi=PLOT_DPI); plt.close(); logger.info(f"Plot saved: {filepath}"); return True
    except Exception as e: logger.error(f"Failed to save plot {filepath}: {e}"); plt.close(); return False

def plot_sample_composition_by_species(composition_data: Dict[str, Counter], filepath: str, top_n: int = 15):
    if not PLOTTING_AVAILABLE or not PANDAS_AVAILABLE or not composition_data:
        logger.debug("Skipping sample composition plot: plotting/pandas unavailable or no data.")
        return False
    plot_df_list = []
    # This plot focuses on 'annotated' count per species. 'random' is not per-species.
    for species, counts in composition_data.items():
        if "annotated" in counts and counts["annotated"] > 0: # Only include if annotated samples exist
            plot_df_list.append({'Species': species, 'Category': "Annotated", 'Count': counts["annotated"]})
    
    if not plot_df_list: 
        logger.debug("Skipping sample composition plot: No 'annotated' data for DataFrame."); return False
    df = pd.DataFrame(plot_df_list)
    if df.empty:
        logger.debug("Skipping sample composition plot: DataFrame empty after processing."); return False

    species_totals = df.groupby('Species')['Count'].sum().nlargest(top_n).index
    df_filtered = df[df['Species'].isin(species_totals)]
    if df_filtered.empty: 
        logger.debug("Skipping sample composition plot: No data after filtering for top N species."); return False
    
    # Pivot table to ensure species are rows and 'Annotated' is a column
    df_pivot = df_filtered.pivot_table(index='Species', columns='Category', values='Count', fill_value=0).fillna(0)
    df_pivot = df_pivot.reindex(species_totals) # Ensure order of top N species
    
    if 'Annotated' not in df_pivot.columns: 
        df_pivot['Annotated'] = 0 # Add column if it somehow got lost (e.g. no annotated samples at all)
    
    df_to_plot = df_pivot[['Annotated']] # Select only the 'Annotated' column for plotting

    fig_width = max(10, len(species_totals) * 0.7)
    df_to_plot.plot(kind='bar', stacked=False, figsize=(fig_width, 7), colormap="Accent") # stacked=False for single bar
    plt.title(f'Annotated Sample Count by Species (Top {len(species_totals)} Species)')
    plt.xlabel('Species'); plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right', fontsize=min(10, 90/len(species_totals) if len(species_totals)>10 else 9))
    plt.legend(title='Sample Category') # Will just show 'Annotated'
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    try: plt.savefig(filepath, dpi=PLOT_DPI); plt.close(); logger.info(f"Plot saved: {filepath}"); return True
    except Exception as e: logger.error(f"Failed to save plot {filepath}: {e}"); plt.close(); return False
# --- End Plotting Functions ---

def manage_file_inputs(fasta_patterns: List[str], gff_patterns: List[str], 
                      explicit_pairs: Optional[List[List[str]]]) -> Dict[str, Dict[str, Any]]:
    """Simplified file input management"""
    species_data = defaultdict(lambda: {'fasta': None, 'gffs': []})
    processed_fastas = set()
    
    # Process explicit pairs first
    if explicit_pairs:
        for species_key, fasta_path, gff_path_str in explicit_pairs: 
            fasta_abs = os.path.abspath(fasta_path)
            if not os.path.exists(fasta_abs):
                logger.warning(f"FASTA not found for explicit pair '{species_key}': {fasta_path}")
                continue
            
            if species_data[species_key]['fasta'] and species_data[species_key]['fasta'] != fasta_abs:
                 logger.warning(f"Species '{species_key}' already has FASTA '{species_data[species_key]['fasta']}'. Overwriting with '{fasta_abs}' due to explicit pair.")

            species_data[species_key]['fasta'] = fasta_abs
            processed_fastas.add(fasta_abs)
            
            if gff_path_str and gff_path_str.lower() != 'none':
                gff_abs = os.path.abspath(gff_path_str)
                if os.path.exists(gff_abs):
                    if gff_abs not in species_data[species_key]['gffs']: # Avoid duplicates
                        species_data[species_key]['gffs'].append(gff_abs)
                else:
                    logger.warning(f"GFF file specified in explicit pair for '{species_key}' not found: {gff_path_str}")
            logger.info(f"Processed explicit pair: Species='{species_key}', FASTA='{fasta_abs}', GFF(s)='{species_data[species_key]['gffs']}'")

    
    # Find all files matching patterns (absolute paths)
    all_fastas_glob = {os.path.abspath(f) for pattern in fasta_patterns 
                  for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)}
    all_gffs_glob = {os.path.abspath(f) for pattern in (gff_patterns or []) 
                for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)}
    
    # Process remaining FASTAs found by glob patterns
    for fasta_path_glob in all_fastas_glob - processed_fastas: # Use renamed var
        species_auto = get_species_from_filename(fasta_path_glob) # Renamed
        
        # Handle cases where auto-derived species name might already exist from explicit pairs
        # or from another globbed FASTA that derived the same species name
        final_species_key = species_auto
        idx = 0
        while species_data[final_species_key]['fasta'] is not None and species_data[final_species_key]['fasta'] != fasta_path_glob:
            idx += 1
            final_species_key = f"{species_auto}_{idx}"
            if idx > 50: # Safety break
                 logger.error(f"Too many collisions for species name '{species_auto}' derived from {fasta_path_glob}. Please use explicit --file-pairs or ensure unique derivable names. Skipping this FASTA.")
                 final_species_key = None # Mark to skip
                 break
        
        if final_species_key is None: continue # Skip if safety break was hit

        species_data[final_species_key]['fasta'] = fasta_path_glob
        
        # Auto-match GFFs from globbed GFFs to this FASTA
        fasta_basename_lower = os.path.basename(fasta_path_glob).lower()
        # Robust prefix: remove common extensions then take first part
        fasta_prefix_candidates = [
            re.sub(r'(\.fasta|\.fa|\.fna|\.genome|\.assembly|\.dna)(\.gz)?$', '', fasta_basename_lower).split('.')[0],
            re.sub(r'(\.fasta|\.fa|\.fna|\.genome|\.assembly|\.dna)(\.gz)?$', '', fasta_basename_lower) # Full name without extension
        ]
        
        matched_gffs_for_this_fasta = []
        for gff_path_glob in all_gffs_glob: # Use renamed var
            gff_basename_lower = os.path.basename(gff_path_glob).lower()
            # Check if any FASTA prefix candidate is in GFF name, or species name is in GFF name
            # More lenient matching:
            found_match = False
            for prefix_cand in fasta_prefix_candidates:
                if prefix_cand and prefix_cand in gff_basename_lower: # prefix_cand might be empty if filename starts with .
                    found_match = True; break
            if not found_match and final_species_key.lower() in gff_basename_lower:
                found_match = True
            
            if found_match:
                if gff_path_glob not in species_data[final_species_key]['gffs']: # Avoid duplicates
                    matched_gffs_for_this_fasta.append(gff_path_glob)

        if matched_gffs_for_this_fasta:
            species_data[final_species_key]['gffs'].extend(matched_gffs_for_this_fasta)
            logger.info(f"Auto-matched FASTA '{fasta_path_glob}' (Key: '{final_species_key}') with GFF(s): {matched_gffs_for_this_fasta}")
        elif all_gffs_glob: # Only log if GFFs were actually searched for
             logger.info(f"No GFFs auto-matched for FASTA '{fasta_path_glob}' (Key: '{final_species_key}').")

    
    # Filter out any entries that didn't actually get a FASTA path (e.g., due to errors or collisions)
    return {k: v for k, v in species_data.items() if v['fasta']}

def prepare_fasta_for_processing(fasta_path: str, temp_dir: Optional[str] = None) -> str:
    """Prepare FASTA file for processing, handling gzipped files"""
    if not fasta_path.endswith('.gz'):
        return fasta_path # Already uncompressed or bgzipped and readable by pyfaidx
    
    # Try to open with pyfaidx first (it handles bgzipped files directly)
    try:
        # logger.info(f"Testing if {fasta_path} is bgzipped and directly usable by pyfaidx...")
        fa_test = Fasta(fasta_path, sequence_always_upper=True)
        # Check if we can actually fetch a contig. Some .gz might open but not be valid BGZF.
        if fa_test.keys(): # If there are contigs
            _ = fa_test[list(fa_test.keys())[0]][:1].seq # Try to fetch 1 base
            fa_test.close()
            logger.info(f"FASTA {fasta_path} appears to be bgzipped and usable directly by pyfaidx.")
            return fasta_path
        else: # No keys, probably not a valid FASTA
            fa_test.close()
            logger.info(f"FASTA {fasta_path} opened by pyfaidx but has no contigs. Will attempt decompression if plain gzip.")
            # Fall through to decompression logic
    except (FastaIndexingError, FetchError, RuntimeError) as e: # Common errors if not bgzipped
        logger.info(f"Direct pyfaidx open for {fasta_path} failed (likely plain gzip or corrupted): {e}. Will attempt decompression.")
        # Fall through to decompression logic
    except Exception as e_unknown: # Other unexpected errors
         logger.warning(f"Unexpected error trying to test {fasta_path} with pyfaidx: {e_unknown}. Will attempt decompression.")
         # Fall through


    # If we reach here, it's either plain gzip or pyfaidx failed for other reasons; attempt decompression.
    logger.info(f"Attempting to decompress plain gzipped file {fasta_path}...")
    
    # Create a more specific temporary directory if a base temp_dir is provided
    final_temp_dir_for_this_file = temp_dir
    if temp_dir:
        # Create a subdirectory within the user-provided temp_dir for this specific file
        # to avoid potential name clashes if multiple gzipped files have the same decompressed name.
        safe_basename = re.sub(r'[^a-zA-Z0-9._-]', '_', os.path.basename(fasta_path))
        per_file_temp_subdir = os.path.join(temp_dir, f"decompressed_{safe_basename}")
        os.makedirs(per_file_temp_subdir, exist_ok=True)
        final_temp_dir_for_this_file = per_file_temp_subdir
    
    # Generate a unique name for the decompressed file
    # Remove .gz, ensure .fa suffix
    base_name = os.path.basename(fasta_path)
    if base_name.endswith(".gz"):
        base_name = base_name[:-3]
    if not (base_name.endswith(".fa") or base_name.endswith(".fasta") or base_name.endswith(".fna")):
        base_name += ".fa"

    # Use NamedTemporaryFile if no specific temp_dir, otherwise construct path in final_temp_dir_for_this_file
    if final_temp_dir_for_this_file:
        temp_fasta_path = os.path.join(final_temp_dir_for_this_file, base_name)
        # Write to this path
        decompression_start_time = time.time()
        try:
            with gzip.open(fasta_path, 'rb') as f_in, open(temp_fasta_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out, length=16*1024*1024) # 16MB buffer
            logger.info(f"Decompressed {fasta_path} to {temp_fasta_path} in {time.time() - decompression_start_time:.2f}s")
            return temp_fasta_path
        except Exception as e_decomp:
            logger.error(f"Failed to decompress {fasta_path} to {temp_fasta_path}: {e_decomp}")
            if os.path.exists(temp_fasta_path): os.unlink(temp_fasta_path) # Clean up partial file
            raise # Re-raise to indicate failure
    else: # Fallback to system's default temp location if no temp_dir provided
        temp_file_obj = tempfile.NamedTemporaryFile(
            suffix='.fa', delete=False, 
            dir=None, # System default temp
            prefix='temp_fasta_decompressed_'
        )
        decompression_start_time = time.time()
        try:
            with gzip.open(fasta_path, 'rb') as f_in:
                shutil.copyfileobj(f_in, temp_file_obj, length=16*1024*1024)
            temp_file_obj.close() # Close file handle before returning path
            logger.info(f"Decompressed {fasta_path} to {temp_file_obj.name} in {time.time() - decompression_start_time:.2f}s")
            return temp_file_obj.name
        except Exception as e_decomp_sys_tmp:
            logger.error(f"Failed to decompress {fasta_path} to system temp file {temp_file_obj.name}: {e_decomp_sys_tmp}")
            os.unlink(temp_file_obj.name) # Clean up
            raise


def generate_report(report_data: Dict[str, Any], report_dir: str, plot_paths: Dict[str, Tuple[Optional[str], str]], pandoc_path: str):
    """Generate markdown report similar to genome_sampler6.py's report"""
    report_md_path = os.path.join(report_dir, "sampling_summary_report.md") # Ensure consistent name
    
    with open(report_md_path, "w", encoding='utf-8') as f_rep: # Use f_rep for clarity
        f_rep.write(f"# Genome Sampler Report\n\n")
        f_rep.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_rep.write(f"Output FASTA: `{os.path.abspath(report_data['args']['output_fasta'])}`\n\n")
        
        f_rep.write("## Configuration Summary\n\n")
        f_rep.write("| Parameter | Value |\n|---|---|\n")
        for k_arg, v_arg in sorted(report_data['args'].items()): 
            escaped_value = str(v_arg).replace('|', '\\|')
            f_rep.write(f"| `{k_arg}` | `{escaped_value}` |\n")
        
        f_rep.write("\n## Runtime Summary\n\n")
        f_rep.write("| Stage | Duration (s) |\n|---|---:|\n")
        for stage_name, duration_s in report_data['timings'].items(): 
            f_rep.write(f"| {stage_name.replace('_', ' ').capitalize()} | {duration_s:.2f} |\n")
        
        f_rep.write("\n## File Processing & GFF Statistics\n\n")
        num_species_processed = len(report_data['species_stats'])
        f_rep.write(f"- Species processed: {num_species_processed}\n")
        
        gff_summary = report_data.get('gff_stats', Counter())
        f_rep.write(f"- Total GFF lines processed across all files: {gff_summary.get('total_lines_processed', 0)}\n")
        f_rep.write(f"- Total GFF features parsed (passing initial GFF parser filters): {gff_summary.get('total_features_kept', 0)}\n")
        
        f_rep.write("\n### GFF Parsing Details (Aggregated):\n")
        for stat_key, stat_val in sorted(gff_summary.items()):
            if stat_key not in ['total_features_kept', 'total_lines_processed', 'raw_source_column', 'raw_feat_type_unfiltered', 'type'] and not stat_key.startswith("raw_") and not stat_key.startswith("type:"): # Avoid redundant detailed counts here
                 f_rep.write(f"  - {stat_key.replace('_',' ').capitalize()}: {stat_val}\n")
        
        f_rep.write("\n### Per-Species GFF Candidate & Selection Summary\n\n")
        f_rep.write("| Species | Initial GFF Candidates (Parser Output) | Valid Candidates (Contig in FASTA) | Annotated Samples Selected |\n|---|---:|---:|---:|\n")
        for species, stats in sorted(report_data['species_stats'].items()):
            f_rep.write(f"| {species} | {stats.get('candidates_initial',0)} | {stats.get('candidates_valid',0)} | {stats.get('samples_selected_annotated',0)} |\n")
        
        f_rep.write("\n## Final Sample Statistics\n\n")
        f_rep.write(f"- Total samples written: {report_data['total_samples_written']}\n")
        f_rep.write("- Samples by Category:\n")
        final_counts = report_data['final_sample_counts']
        for cat_name in sorted(final_counts.keys()): # Iterate defined categories
            count_cat = final_counts[cat_name]
            f_rep.write(f"  - **{cat_name.capitalize()}**: {count_cat}\n")
            
            cat_lengths = report_data["length_distributions_by_category"].get(cat_name, [])
            cat_gcs = [g for g in report_data["gc_content_distributions_by_category"].get(cat_name, []) if g is not None]
            cat_entropies = [e for e in report_data["entropy_distributions_by_category"].get(cat_name, []) if e is not None]

            if cat_lengths:
                L_stats = f"Min={min(cat_lengths)}, Max={max(cat_lengths)}, Avg={sum(cat_lengths)/len(cat_lengths):.1f}, Median={np.median(cat_lengths):.1f}"
                f_rep.write(f"    - Lengths (bp): {L_stats}\n")
            if cat_gcs:
                GC_stats = f"Min={min(cat_gcs):.1f}, Max={max(cat_gcs):.1f}, Avg={sum(cat_gcs)/len(cat_gcs):.1f}, Median={np.median(cat_gcs):.1f}"
                f_rep.write(f"    - GC Content (%): {GC_stats}\n")
            if cat_entropies:
                ENT_stats = f"Min={min(cat_entropies):.2f}, Max={max(cat_entropies):.2f}, Avg={sum(cat_entropies)/len(cat_entropies):.2f}, Median={np.median(cat_entropies):.2f}"
                f_rep.write(f"    - Shannon Entropy (bits): {ENT_stats}\n")

        f_rep.write("\n### Species Distribution in Final Annotated Sample (Top 20 by count)\n\n| Species | Count | Avg GC (%) | Avg Entropy (bits) |\n|---|---:|---:|---:|\n")
        for sp_final, ct_final in report_data["species_distribution_in_final_samples"].most_common(20):
            sp_gcs_final = [g for g in report_data["gc_by_species_plotdata"].get(sp_final, []) if g is not None]
            sp_ents_final = [e for e in report_data["entropy_by_species_plotdata"].get(sp_final, []) if e is not None]
            avg_gc_str = f"{sum(sp_gcs_final)/len(sp_gcs_final):.1f}" if sp_gcs_final else "N/A"
            avg_ent_str = f"{sum(sp_ents_final)/len(sp_ents_final):.2f}" if sp_ents_final else "N/A"
            f_rep.write(f"| {sp_final} | {ct_final} | {avg_gc_str} | {avg_ent_str} |\n")

        f_rep.write("\n### Biotype Distribution in Final Annotated Samples (Top 20 by count)\n\n| Biotype | Count | Avg GC (%) | Avg Entropy (bits) |\n|---|---:|---:|---:|\n")
        for bio_final, ct_bio_final in report_data["biotype_distribution_in_final_annotated_samples"].most_common(20):
            bio_gcs_final = [g for g in report_data["gc_by_biotype_plotdata"].get(bio_final, []) if g is not None]
            bio_ents_final = [e for e in report_data["entropy_by_biotype_plotdata"].get(bio_final, []) if e is not None]
            avg_gc_bio_str = f"{sum(bio_gcs_final)/len(bio_gcs_final):.1f}" if bio_gcs_final else "N/A"
            avg_ent_bio_str = f"{sum(bio_ents_final)/len(bio_ents_final):.2f}" if bio_ents_final else "N/A"
            f_rep.write(f"| {bio_final} | {ct_bio_final} | {avg_gc_bio_str} | {avg_ent_bio_str} |\n")
        
        if PLOTTING_AVAILABLE and plot_paths:
            f_rep.write("\n## Plots\n\n")
            # Define a preferred order for plots in the report
            plot_display_order = [
                'overall_len_dist', 'annotated_len_dist', 'random_len_dist', 'unknown_len_dist', 'len_dist_comparison',
                'annotated_orig_len_dist', 'annotated_orig_vs_sampled_len_scatter',
                'gc_dist_by_sample_category', 'entropy_dist_by_sample_category',
                'len_vs_gc_all_colored', 'len_vs_entropy_all_colored',
                'contig_mismatch_summary', 'annotation_coverage_species',
                'sample_composition_by_species',
                'species_dist_final', 'biotype_dist_final', 'kept_gff_types_from_biotype_dist',
                'gc_dist_by_species', 'entropy_dist_by_species', 'gc_dist_by_biotype', 'entropy_dist_by_biotype',
                'raw_gff_sources_dist', 'raw_gff_types_dist'
            ]
            for key_plot in plot_display_order:
                if key_plot in plot_paths and plot_paths[key_plot][0] is not None:
                    filename, title = plot_paths[key_plot]
                    # Ensure filename is just the basename for relative linking in Markdown
                    f_rep.write(f"### {title}\n\n")
                    f_rep.write(f"![{title}]({os.path.basename(filename)})\n\n") 
    
    logger.info(f"Markdown report saved to: {report_md_path}")
    
    try:
        pdf_path = os.path.join(report_dir, "sampling_summary_report.pdf")
        if shutil.which(pandoc_path): # Check if pandoc is in PATH
            # Use cwd=report_dir so pandoc can find images with relative paths
            process = subprocess.run([
                pandoc_path, os.path.basename(report_md_path), '-o', os.path.basename(pdf_path),
                '--pdf-engine=xelatex', '-V', 'geometry:margin=0.75in', '--toc'
            ], check=True, capture_output=True, text=True, cwd=report_dir, timeout=120)
            logger.info(f"PDF report saved to: {pdf_path}")
        else:
            logger.warning(f"Pandoc executable '{pandoc_path}' not found in PATH. Skipping PDF generation.")
    except subprocess.TimeoutExpired:
        logger.error(f"Pandoc PDF generation timed out for {report_md_path}.")
    except subprocess.CalledProcessError as e_pandoc:
        logger.error(f"Pandoc PDF generation failed. Return code: {e_pandoc.returncode}")
        logger.error(f"Pandoc stdout: {e_pandoc.stdout}")
        logger.error(f"Pandoc stderr: {e_pandoc.stderr}")
        logger.warning("PDF generation failed. Check pandoc installation and LaTeX dependencies (like xelatex, titling, and required fonts).")
    except FileNotFoundError: # If pandoc_path itself is invalid and not caught by shutil.which
         logger.warning(f"Pandoc executable '{pandoc_path}' not found. Skipping PDF generation.")
    except Exception as e_general_pandoc:
        logger.warning(f"PDF generation failed due to an unexpected error: {e_general_pandoc}")


def main():
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    root_logger = logging.getLogger()
    # Remove any existing handlers to prevent duplicate messages if script is re-run in same session
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    # Default level, will be updated by args
    root_logger.setLevel(logging.INFO) 

    # Now that logger is configured, PANDAS_AVAILABLE/PLOTTING_AVAILABLE warnings can use it
    # This check needs to be after basicConfig or manual logger setup
    if 'warning_msg' in globals() and warning_msg: # Check if PANDAS/Plotting warning was set
        logger.warning(warning_msg) # Use the now-configured logger

    parser = argparse.ArgumentParser(
        description="Sample sequences from FASTA files with optional GFF3 annotation guidance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument("--fasta_patterns", nargs='+', required=True, help="Glob pattern(s) for input FASTA files (e.g., 'data/*.fa.gz').")
    io_group.add_argument("--gff_patterns", nargs='+', help="Glob pattern(s) for input GFF3 files (e.g., 'data/*.gff.gz').")
    io_group.add_argument("--file_pairs", nargs=3, action='append', metavar=('SPECIES_KEY', 'FASTA_PATH', 'GFF_PATH'), 
                          help="Explicitly define a species, its FASTA, and GFF. GFF_PATH can be 'None'. Repeat for multiple pairs.")
    io_group.add_argument("--output_fasta", required=True, help="Path for the final sampled FASTA output.")
    io_group.add_argument("--report_dir", default="sampler_report", help="Directory for reports and plots.")
    io_group.add_argument("--allow_plain_gzip_fallback", action="store_true", help="If a .gz FASTA is not bgzipped, attempt to decompress it to a temporary file. Requires free disk space.")
    io_group.add_argument("--temp_dir", default=None, help="Base directory for temporary files (e.g., decompressed FASTAs). If None, system default is used.")
    
    sampling_group = parser.add_argument_group('Sampling Parameters')
    sampling_group.add_argument("--total_samples", type=int, default=10000, help="Total number of sequences to sample.")
    sampling_group.add_argument("--annotated_proportion", type=float, default=0.8, help="Proportion of samples to be derived from GFF annotations.")
    sampling_group.add_argument("--random_proportion", type=float, default=0.2, help="Proportion of samples to be randomly generated DNA.")
    sampling_group.add_argument("--min_len", type=int, default=DEFAULT_MIN_LEN, help="Minimum length for sampled sequences.")
    sampling_group.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN, help="Maximum length for sampled sequences.")
    sampling_group.add_argument("--truncation_strategy", default=DEFAULT_TRUNCATION_STRATEGY,
                               choices=["start", "center", "end", "random_segment", "none"],
                               help="Strategy for handling features/sequences longer than max_len. 'none' will skip them if they exceed max_len.")
    
    gff_group = parser.add_argument_group('GFF Filtering')
    gff_group.add_argument("--annotation_confidence_level", default=DEFAULT_CONFIDENCE_LEVEL,
                          choices=list(CONFIDENCE_LEVELS.keys()) + ["custom"], help="Predefined confidence level for GFF source/term filtering.")
    gff_group.add_argument("--custom_trusted_gff_sources", type=str, help="Comma-separated trusted GFF sources (used if confidence is 'custom' or to augment predefined).")
    gff_group.add_argument("--custom_low_confidence_terms_gff", type=str, help="Comma-separated regex terms indicating low confidence (used if confidence is 'custom' or to augment predefined).")
    gff_group.add_argument("--target_gff_feature_types", type=str, help="Comma-separated GFF feature types to target (e.g., 'gene,mRNA,CDS'). If None, most common types are processed.")
    gff_group.add_argument("--description_gff_attributes", default=DEFAULT_DESCRIPTION_ATTRIBUTES,
                           help="Comma-separated GFF attribute keys (in order of preference) to use for sequence descriptions.")
    
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    misc_group.add_argument("--log_level", default="INFO",
                           choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")
    misc_group.add_argument("--num_workers", type=int, default=1, help="Number of workers for parallel GFF parsing (currently not implemented, uses 1).") # Kept for compatibility, but sampler is serial
    misc_group.add_argument("--pandoc_path", default="pandoc", help="Path to Pandoc executable for PDF report generation.")
    
    args = parser.parse_args()
    
    # Update logger level based on args
    logger.setLevel(getattr(logging, args.log_level.upper()))
    # Also set level for the root logger if it was separately obtained
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    start_time = time.time()
    stage_times = {}
    
    os.makedirs(args.report_dir, exist_ok=True)
    # Create a root temporary directory for this run if user specified a base temp_dir or if fallback is allowed
    # This root_temp_dir_for_run will house subdirectories for each decompressed file.
    root_temp_dir_for_run = None
    temp_files_to_cleanup = [] # Store paths of files created in temp locations

    if args.allow_plain_gzip_fallback:
        if args.temp_dir:
            # User specified a base temp directory
            # Create a unique sub-directory within it for this run
            run_specific_temp_path = os.path.join(args.temp_dir, f"genome_sampler_run_{os.getpid()}_{int(time.time())}")
            try:
                os.makedirs(run_specific_temp_path, exist_ok=True)
                root_temp_dir_for_run = run_specific_temp_path
                logger.info(f"Using run-specific temporary directory: {root_temp_dir_for_run}")
            except OSError as e:
                logger.error(f"Could not create run-specific temporary directory at {run_specific_temp_path}: {e}. Fallback to system default for temp files if needed.")
                # root_temp_dir_for_run remains None, prepare_fasta_for_processing will use system default temp.
        else:
            # No user temp_dir, but fallback is allowed. prepare_fasta_for_processing will use system default.
            # root_temp_dir_for_run can remain None.
            pass


    # Normalize proportions
    total_prop = args.annotated_proportion + args.random_proportion
    if not math.isclose(total_prop, 1.0) or total_prop == 0: # Handle sum is zero
        logger.warning(f"Proportions sum to {total_prop:.3f}. Normalizing or resetting to defaults (0.8 anno, 0.2 random).")
        if total_prop == 0: # Avoid division by zero if both are 0
            args.annotated_proportion = 0.8
            args.random_proportion = 0.2
        else:
            args.annotated_proportion /= total_prop
            args.random_proportion /= total_prop
    
    num_annotated = int(args.total_samples * args.annotated_proportion)
    # Ensure random gets at least what's left, or recalculate if total_samples is the strict target
    num_random = args.total_samples - num_annotated 
    if num_random < 0: num_random = 0 # Should not happen with normalization

    logger.info(f"Target samples: Total={args.total_samples}, Annotated={num_annotated}, Random={num_random}")
    
    # Prepare GFF parsing parameters
    if args.annotation_confidence_level == "custom":
        trusted_sources = set(s.strip() for s in args.custom_trusted_gff_sources.split(',')) if args.custom_trusted_gff_sources else None
        low_conf_terms = [s.strip() for s in args.custom_low_confidence_terms_gff.split(',')] if args.custom_low_confidence_terms_gff else []
    else:
        conf_settings = CONFIDENCE_LEVELS[args.annotation_confidence_level]
        trusted_sources = conf_settings["sources"].copy() if conf_settings["sources"] else None # Ensure mutable set or None
        low_conf_terms = list(conf_settings["low_confidence_terms"])
        # Augment with custom if provided
        if args.custom_trusted_gff_sources and trusted_sources is not None:
            trusted_sources.update(s.strip() for s in args.custom_trusted_gff_sources.split(','))
        elif args.custom_trusted_gff_sources: # if predefined was None
             trusted_sources = set(s.strip() for s in args.custom_trusted_gff_sources.split(','))
        if args.custom_low_confidence_terms_gff:
            low_conf_terms.extend(s.strip() for s in args.custom_low_confidence_terms_gff.split(','))
            low_conf_terms = sorted(list(set(low_conf_terms))) # Deduplicate and sort

    low_conf_regex_str = r'(' + '|'.join(re.escape(term) for term in low_conf_terms) + r')' if low_conf_terms else r'(?!)' # Avoid empty regex part
    low_conf_regex = re.compile(low_conf_regex_str, re.IGNORECASE)
    
    target_types = {t.strip().lower() for t in args.target_gff_feature_types.split(',')} if args.target_gff_feature_types else None
    desc_attrs = [a.strip() for a in args.description_gff_attributes.split(',')]
    
    # GFF parser initial max_len depends on truncation strategy for efficiency
    # If 'none', parser filters strictly by max_len. Otherwise, parser allows longer, sampler truncates.
    gff_parser_initial_max_len = args.max_len if args.truncation_strategy == "none" else float('inf')
    gff_parser_params = {
        'trusted_sources': trusted_sources,
        'low_conf_regex': low_conf_regex,
        'desc_attrs': desc_attrs,
        'target_types': target_types,
        'min_len': args.min_len, 
        'max_len': gff_parser_initial_max_len 
    }
    
    stage_start_time_input = time.time() # Renamed
    species_data = manage_file_inputs(args.fasta_patterns, args.gff_patterns or [], args.file_pairs)
    stage_times['file_input'] = time.time() - stage_start_time_input
    
    if not species_data:
        logger.critical("No valid FASTA files found or mapped. Exiting.")
        sys.exit(1)
    
    logger.info(f"Found {len(species_data)} species to process based on input parameters.")
    
    for species, file_info_dict in species_data.items(): # Renamed
        if args.allow_plain_gzip_fallback and file_info_dict['fasta'].endswith('.gz'):
            try:
                new_fasta_path = prepare_fasta_for_processing(file_info_dict['fasta'], root_temp_dir_for_run)
                if new_fasta_path != file_info_dict['fasta']: 
                    temp_files_to_cleanup.append(new_fasta_path)
                    file_info_dict['fasta'] = new_fasta_path 
            except Exception as e_prep:
                logger.error(f"Failed to prepare FASTA {file_info_dict['fasta']} for species {species}: {e_prep}. This species might be skipped or cause errors.")

    stage_start_time_processing = time.time() # Renamed
    logger.info("=== Starting main GFF processing and sample selection ===")
    processing_result = process_species_files(
        species_data, gff_parser_params,
        num_annotated, num_random, 
        args.min_len, args.max_len, args.truncation_strategy
    )
    stage_times['gff_processing_and_task_selection'] = time.time() - stage_start_time_processing
    logger.info(f"GFF processing and sample task selection completed in {stage_times['gff_processing_and_task_selection']:.2f}s")
    
    stage_start_time_writing = time.time() # Renamed
    logger.info("=== Starting sequence extraction and writing ===")
    write_result = write_samples(
        args.output_fasta, species_data,
        processing_result['annotated_samples'],
        processing_result['random_samples'],
        args.min_len, args.max_len, args.truncation_strategy
    )
    stage_times['sequence_extraction_and_writing'] = time.time() - stage_start_time_writing
    logger.info(f"Sequence extraction and writing completed in {stage_times['sequence_extraction_and_writing']:.2f}s")
    
    report_data = {
        'args': vars(args),
        'timings': stage_times,
        'species_stats': processing_result['species_stats'],
        'gff_stats': processing_result['gff_stats'],
        'contig_mismatch_summary_per_species': processing_result['contig_mismatches'],
        'annotation_coverage_plotdata': processing_result['species_coverage_stats'], 
        'sampling_process_stats_summary': write_result['sampling_stats'], 
        'total_samples_written': write_result['sampling_stats'].get('total_written', 0),
        'final_sample_counts': Counter(), 
        'length_distributions_by_category': defaultdict(list),
        'gc_content_distributions_by_category': defaultdict(list),
        'entropy_distributions_by_category': defaultdict(list),
        'species_distribution_in_final_samples': Counter(),
        'biotype_distribution_in_final_annotated_samples': Counter(),
        'kept_gff_feature_types_from_biotype_plotdata': Counter(), 
        'final_annotated_original_lengths': write_result['annotated_lengths_orig'],
        'final_annotated_sampled_lengths': write_result['annotated_lengths_sampled'],
        'all_final_lengths_gc_categories_plotdata': {'lengths': [], 'gcs': [], 'entropies': [], 'categories': []},
        'gc_by_sample_category_plotdata': defaultdict(list),
        'entropy_by_sample_category_plotdata': defaultdict(list),
        'gc_by_species_plotdata': defaultdict(list),
        'entropy_by_species_plotdata': defaultdict(list),
        'gc_by_biotype_plotdata': defaultdict(list),
        'entropy_by_biotype_plotdata': defaultdict(list),
        'sample_composition_by_species_plotdata': defaultdict(Counter),
    }
    
    for header, category, gc, entropy, sampled_len, original_len, species_name, biotype_name in write_result['metadata']:
        report_data['final_sample_counts'][category] += 1
        report_data['length_distributions_by_category'][category].append(sampled_len)
        report_data['length_distributions_by_category']['overall'].append(sampled_len)
        report_data['all_final_lengths_gc_categories_plotdata']['lengths'].append(sampled_len)
        report_data['all_final_lengths_gc_categories_plotdata']['gcs'].append(gc if gc is not None else float('nan'))
        report_data['all_final_lengths_gc_categories_plotdata']['entropies'].append(entropy if entropy is not None else float('nan'))
        report_data['all_final_lengths_gc_categories_plotdata']['categories'].append(category)
        if gc is not None:
            report_data['gc_content_distributions_by_category'][category].append(gc)
            report_data['gc_content_distributions_by_category']['overall'].append(gc)
            report_data['gc_by_sample_category_plotdata'][category].append(gc)
        if entropy is not None:
            report_data['entropy_distributions_by_category'][category].append(entropy)
            report_data['entropy_distributions_by_category']['overall'].append(entropy)
            report_data['entropy_by_sample_category_plotdata'][category].append(entropy)
        if category == "annotated":
            if species_name != "random": 
                report_data['species_distribution_in_final_samples'][species_name] += 1
                report_data['sample_composition_by_species_plotdata'][species_name][category] += 1
                if gc is not None: report_data['gc_by_species_plotdata'][species_name].append(gc)
                if entropy is not None: report_data['entropy_by_species_plotdata'][species_name].append(entropy)
            if biotype_name != "random_dna": 
                report_data['biotype_distribution_in_final_annotated_samples'][biotype_name] += 1
                if gc is not None: report_data['gc_by_biotype_plotdata'][biotype_name].append(gc)
                if entropy is not None: report_data['entropy_by_biotype_plotdata'][biotype_name].append(entropy)
                if biotype_name not in ["unknown", "N/A", "random_dna"]: 
                     report_data['kept_gff_feature_types_from_biotype_plotdata'][biotype_name] += 1
    
    stage_times['total_script_runtime'] = time.time() - start_time
    
    plot_paths = {} 
    if PLOTTING_AVAILABLE:
        logger.info("Generating plots...")
        stage_start_time_plotting = time.time() 
        
        raw_gff_sources_plot_data = Counter({
            k.split("raw_source_column:",1)[1]: v 
            for k,v in report_data["gff_stats"].items() 
            if k.startswith("raw_source_column:")
        })
        raw_gff_types_plot_data = Counter({
            k.split("raw_feat_type_unfiltered:",1)[1]: v 
            for k,v in report_data["gff_stats"].items() 
            if k.startswith("raw_feat_type_unfiltered:")
        })

        plot_configs = {
            'overall_len_dist': (plot_length_distribution, [report_data['length_distributions_by_category'].get('overall',[]), "Overall Sampled Sequence Lengths", os.path.join(args.report_dir, "overall_len_dist_length_distribution.png")], "Overall Sampled Length Distribution"),
            'annotated_len_dist': (plot_length_distribution, [report_data['length_distributions_by_category'].get('annotated',[]), "Annotated Sample Lengths", os.path.join(args.report_dir, "annotated_len_dist_length_distribution.png")], "Annotated Sample Lengths Distribution"),
            'random_len_dist': (plot_length_distribution, [report_data['length_distributions_by_category'].get('random',[]), "Random Sample Lengths", os.path.join(args.report_dir, "random_len_dist_length_distribution.png")], "Random Sample Lengths Distribution"),
            'unknown_len_dist': (plot_length_distribution, [report_data['length_distributions_by_category'].get('unknown',[]), "Unknown Sample Lengths", os.path.join(args.report_dir, "unknown_len_dist_length_distribution.png")], "Unknown Sample Lengths Distribution"), 
            'len_dist_comparison': (plot_length_distribution_comparison, [report_data['length_distributions_by_category'], "Sample Length Distributions by Category", os.path.join(args.report_dir, "len_dist_comparison_length_distribution_comparison.png")], "Sampled Length Distributions by Category (Overlay)"),
            'annotated_orig_len_dist': (plot_length_distribution, [report_data['final_annotated_original_lengths'], "Original Lengths of Annotated Features (Pool)", os.path.join(args.report_dir, "annotated_orig_len_dist_length_distribution.png")], "Original Lengths of Annotated Features (Pool)"),
            'annotated_orig_vs_sampled_len_scatter': (plot_scatter_lengths, [report_data['final_annotated_original_lengths'], report_data['final_annotated_sampled_lengths'], "Annotated Features: Original vs. Sampled Lengths", os.path.join(args.report_dir, "annotated_orig_vs_sampled_len_scatter_scatter_lengths.png")], "Annotated: Original vs. Sampled Lengths"),
            'species_dist_final': (plot_categorical_distribution, [report_data['species_distribution_in_final_samples'], "Species Distribution (Final Annotated Samples)", "Species", os.path.join(args.report_dir, "species_dist_final_categorical_distribution.png"), 20], "Species Distribution (Annotated Samples, Top 20)"),
            'biotype_dist_final': (plot_categorical_distribution, [report_data['biotype_distribution_in_final_annotated_samples'], "Biotype Distribution (Final Annotated Samples)", "Biotype", os.path.join(args.report_dir, "biotype_dist_final_categorical_distribution.png"), 20], "Biotype Distribution (Annotated Samples, Top 20)"),
            'kept_gff_types_from_biotype_dist': (plot_categorical_distribution, [report_data['kept_gff_feature_types_from_biotype_plotdata'], "Kept GFF Feature Types (from Biotype in Annotated)", "Feature Type (from Biotype)", os.path.join(args.report_dir, "kept_gff_types_from_biotype_dist_categorical_distribution.png"), 20], "Kept GFF Types (from Biotype, Annotated, Top 20)"),
            'raw_gff_sources_dist': (plot_categorical_distribution, [raw_gff_sources_plot_data, "Raw GFF Sources Encountered (Pre-filter)", "GFF Source", os.path.join(args.report_dir, "raw_gff_sources_dist_categorical_distribution.png"), 25], "Raw GFF Sources (Pre-filter, Top 25)"),
            'raw_gff_types_dist': (plot_categorical_distribution, [raw_gff_types_plot_data, "Raw GFF Feature Types Encountered (Pre-filter)", "GFF Type", os.path.join(args.report_dir, "raw_gff_types_dist_categorical_distribution.png"), 25], "Raw GFF Types (Pre-filter, Top 25)"),
            'gc_dist_by_sample_category': (plot_numerical_dist_by_category, [report_data['gc_by_sample_category_plotdata'], "GC Content (%)", "Sample Category", "GC Content by Sample Category", os.path.join(args.report_dir, "gc_dist_by_sample_category_numerical_dist_by_category.png")], "GC Content by Sample Category"),
            'entropy_dist_by_sample_category': (plot_numerical_dist_by_category, [report_data['entropy_by_sample_category_plotdata'], "Shannon Entropy (bits)", "Sample Category", "Shannon Entropy by Sample Category", os.path.join(args.report_dir, "entropy_dist_by_sample_category_numerical_dist_by_category.png")], "Shannon Entropy by Sample Category"),
            'gc_dist_by_species': (plot_numerical_dist_by_category, [report_data['gc_by_species_plotdata'], "GC Content (%)", "Species (Annotated)", "GC Content by Species (Top 15 Annotated)", os.path.join(args.report_dir, "gc_dist_by_species_numerical_dist_by_category.png"), 15], "GC Content by Species (Annotated, Top 15)"),
            'entropy_dist_by_species': (plot_numerical_dist_by_category, [report_data['entropy_by_species_plotdata'], "Shannon Entropy (bits)", "Species (Annotated)", "Shannon Entropy by Species (Top 15 Annotated)", os.path.join(args.report_dir, "entropy_dist_by_species_numerical_dist_by_category.png"), 15], "Shannon Entropy by Species (Annotated, Top 15)"),
            'gc_dist_by_biotype': (plot_numerical_dist_by_category, [report_data['gc_by_biotype_plotdata'], "GC Content (%)", "Biotype (Annotated)", "GC Content by Biotype (Top 15 Annotated)", os.path.join(args.report_dir, "gc_dist_by_biotype_numerical_dist_by_category.png"), 15], "GC Content by Biotype (Annotated, Top 15)"),
            'entropy_dist_by_biotype': (plot_numerical_dist_by_category, [report_data['entropy_by_biotype_plotdata'], "Shannon Entropy (bits)", "Biotype (Annotated)", "Shannon Entropy by Biotype (Top 15 Annotated)", os.path.join(args.report_dir, "entropy_dist_by_biotype_numerical_dist_by_category.png"), 15], "Shannon Entropy by Biotype (Annotated, Top 15)"),
            'len_vs_gc_all_colored': (plot_2d_density, [report_data['all_final_lengths_gc_categories_plotdata']['lengths'], report_data['all_final_lengths_gc_categories_plotdata']['gcs'], "Sampled Length (bp)", "GC Content (%)", "Length vs GC Content (All Samples)", os.path.join(args.report_dir, "len_vs_gc_all_colored_2d_density.png"), report_data['all_final_lengths_gc_categories_plotdata']['categories']], "Length vs. GC Content (All Samples, Colored)"),
            'len_vs_entropy_all_colored': (plot_2d_density, [report_data['all_final_lengths_gc_categories_plotdata']['lengths'], report_data['all_final_lengths_gc_categories_plotdata']['entropies'], "Sampled Length (bp)", "Shannon Entropy (bits)", "Length vs Entropy (All Samples)", os.path.join(args.report_dir, "len_vs_entropy_all_colored_2d_density.png"), report_data['all_final_lengths_gc_categories_plotdata']['categories']], "Length vs. Entropy (All Samples, Colored)"),
            'contig_mismatch_summary': (plot_contig_mismatch_summary, [report_data['contig_mismatch_summary_per_species'], os.path.join(args.report_dir, "contig_mismatch_summary_plot_contig_mismatch_summary.png")], "GFF/FASTA Contig Concordance"),
            'sample_composition_by_species': (plot_sample_composition_by_species, [report_data['sample_composition_by_species_plotdata'], os.path.join(args.report_dir, "sample_composition_by_species_plot_sample_composition_by_species.png")], "Annotated Sample Composition by Species (Top 15)"),
            'annotation_coverage_species': (plot_annotation_coverage_per_species, [report_data['annotation_coverage_plotdata'], os.path.join(args.report_dir, "annotation_coverage_species_annotation_coverage_per_species.png")], "Proportion of FASTA Contigs with GFF Features"),
        }
        
        for plot_key, (plot_func, plot_args_list_from_config, report_title) in plot_configs.items():
            filepath_arg_found = None
            for arg_item in plot_args_list_from_config:
                if isinstance(arg_item, str) and arg_item.endswith(".png"):
                    filepath_arg_found = arg_item
                    break 
            
            if filepath_arg_found:
                try:
                    plot_func(*plot_args_list_from_config) 
                    plot_paths[plot_key] = (filepath_arg_found, report_title)
                except Exception as e_plot_call:
                     logger.error(f"Error calling plot function for '{plot_key}' with args {plot_args_list_from_config}: {e_plot_call}", exc_info=True)
            else:
                logger.warning(f"Could not determine filepath for plot '{plot_key}' from its arguments: {plot_args_list_from_config}. Skipping storage in plot_paths.")

        stage_times['plotting'] = time.time() - stage_start_time_plotting
        logger.info(f"Plot generation finished in {stage_times['plotting']:.2f} seconds.")
    
    stage_start_time_reporting = time.time() # Renamed
    generate_report(report_data, args.report_dir, plot_paths, args.pandoc_path)
    stage_times['reporting'] = time.time() - stage_start_time_reporting
    
    for temp_file_path in temp_files_to_cleanup:
        try:
            if os.path.isfile(temp_file_path):
                os.unlink(temp_file_path)
                logger.debug(f"Removed temporary file: {temp_file_path}")
            elif os.path.isdir(temp_file_path): 
                shutil.rmtree(temp_file_path)
                logger.debug(f"Removed temporary directory (unexpectedly in file list): {temp_file_path}")
        except Exception as e_cleanup_file:
            logger.warning(f"Failed to remove temporary item {temp_file_path}: {e_cleanup_file}")
    
    if root_temp_dir_for_run and os.path.exists(root_temp_dir_for_run):
        try:
            shutil.rmtree(root_temp_dir_for_run)
            logger.info(f"Removed run-specific temporary directory: {root_temp_dir_for_run}")
        except Exception as e_cleanup_root:
            logger.warning(f"Failed to remove run-specific temporary directory {root_temp_dir_for_run}: {e_cleanup_root}")

    
    logger.info(f"Script completed in {stage_times['total_script_runtime']:.2f} seconds.")
    logger.info(f"Total samples written: {report_data['total_samples_written']}")
    logger.info(f"  - Annotated: {report_data['final_sample_counts'].get('annotated', 0)}")
    logger.info(f"  - Random: {report_data['final_sample_counts'].get('random', 0)}")
    logger.info("Stage timings:")
    for stage_item, duration_item in report_data['timings'].items(): # Renamed
        logger.info(f"  - {stage_item.replace('_', ' ').capitalize()}: {duration_item:.2f}s")

if __name__ == "__main__":
    main()