"""
A module to reconstruct nucleotide sequences with the most parsimonious codons
and classify genetic data based on a fold-change threshold.

This module includes functionality for parsing genotype-phenotype datasets,
computing parsimonious codons to minimize nucleotide substitutions, and
exporting processed data for further analysis. Additionally, it provides
methods for visualizing fold-change distributions.

Classes:
    ReconstructSequence: Handles data parsing, processing, and codon reconstruction.
"""
import random
import csv
from math import log2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde, probplot
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


class ReconstructSequence:

    """
    A script to convert the amino-acid-based Stanford style geno-pheno.dataset.tsv file data into
    an inferred nucleotide sequence file suitable for use with the producer code.
    Note that simplifying assumptions are made in this process, such as the use of the most parsimonious codon
    for each amino acid position - i.e., fewer substitions are preferred first, and transversion mutations are used only
    when a transition mutation to the new codon is not possible. If multiple equally parsimonious codons exist, one is
    chosen randomly. Note that the Stanford database sometimes suggests multiple possible amino acids for a position,
    presumably arising from ambiguity in the underlying DNA sequence. This program just selects the first listed
    possible amino acid.  The reference integrase sequence is hard-coded and taken from strain HXB2 / accession K03455
    """

    def __init__(self, input_path='geno-pheno.dataset.tsv', output_path='geno_pheno_processed.csv', threshold=log2(1.5)):

        self.path = input_path
        self.output_path = output_path
        self.threshold = threshold

        # Standard genetic code (DNA version)
        self.genetic_code = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
            }

        self.transitions = {
            ('A', 'G'), ('G', 'A'),
            ('C', 'T'), ('T', 'C')
            }

        self.ref_integrase = (
            'TTTTTAGATGGAATAGATAAGGCCCAAGATGAACATGAGAAATATCACAGTAATTGGAGAGCAATGGCTAGTGATTTTAACCTGCCACCTGTAGTAGCAAAAGAAA'
            'TAGTAGCCAGCTGTGATAAATGTCAGCTAAAAGGAGAAGCCATGCATGGACAAGTAGACTGTAGTCCAGGAATATGGCAACTAGATTGTACACATTTAGAAGGAAA'
            'AGTTATCCTGGTAGCAGTTCATGTAGCCAGTGGATATATAGAAGCAGAAGTTATTCCAGCAGAAACAGGGCAGGAAACAGCATATTTTCTTTTAAAATTAGCAGGA'
            'AGATGGCCAGTAAAAACAATACATACTGACAATGGCAGCAATTTCACCGGTGCTACGGTTAGGGCCGCCTGTTGGTGGGCGGGAATCAAGCAGGAATTTGGAATTC'
            'CCTACAATCCCCAAAGTCAAGGAGTAGTAGAATCTATGAATAAAGAATTAAAGAAAATTATAGGACAGGTAAGAGATCAGGCTGAACATCTTAAGACAGCAGTACA'
            'AATGGCAGTATTCATCCACAATTTTAAAAGAAAAGGGGGGATTGGGGGGTACAGTGCAGGGGAAAGAATAGTAGACATAATAGCAACAGACATACAAACTAAAGAA'
            'TTACAAAAACAAATTACAAAAATTCAAAATTTTCGGGTTTATTACAGGGACAGCAGAAATCCACTTTGGAAAGGACCAGCAAAGCTCCTCTGGAAAGGTGAAGGGG'
            'CAGTAGTAATACAAGATAATAGTGACATAAAAGTAGTGCCAAGAAGAAAAGCAAAGATCATTAGGGATTATGGAAAACAGATGGCAGGTGATGATTGTGTGGCAAG'
            'TAGACAGGATGAGGAT'
        )

    def parse_tsv(self):
        """
        Parse a tab-delimited file with a header row.
        Yields one dictionary per data row, mapping column name -> value.
        """
        with open(self.path, newline='', encoding='utf-8') as ifh:
            reader = csv.DictReader(ifh, delimiter='\t')
            for row in reader:
                yield row

    def parse_with_position_column(self):
        """
        Returns a list of records, where each record is a dict with:
          - 'metadata': non-position columns
          - 'positions': dict mapping position index (int) -> value (e.g. '-', 'A', 'GR')
        """
        records = []

        for row in self.parse_tsv():
            metadata = {}
            positions = {}

            for k, v in row.items():
                if k.startswith('P'):
                    # P1, P2, ..., P288
                    try:
                        pos = int(k[1:])
                    except ValueError:
                        continue
                    positions[pos] = v
                else:
                    metadata[k] = v

            records.append({
                'metadata': metadata,
                'positions': positions
            })

        return records


    def most_parsimonious_codon(self, target_aa, start_codon):
        """
        Input: target amino acid, starting codon to be transformed
        Returns: (chosen_codon, n_substitutions, n_equally_parsimonious_solutions)
        """

        # All codons encoding the target amino acid
        target_codons = [c for c, aa in self.genetic_code.items() if aa == target_aa]

        results = []

        for codon in target_codons:
            subs = 0
            trans = 0
            for a, b in zip(start_codon, codon):
                if a != b:
                    subs += 1
                    if (a, b) in self.transitions:
                        trans += 1
            results.append((codon, subs, trans))

        # Primary criterion: minimize substitutions
        min_subs = min(r[1] for r in results)
        results = [r for r in results if r[1] == min_subs]

        # Secondary criterion: maximize transitions
        max_trans = max(r[2] for r in results)
        results = [r for r in results if r[2] == max_trans]

        # if more than one possible input_path is equally parsimonious, choose randomly
        chosen = random.choice(results)
        n_paths = len(results)

        return chosen[0], min_subs, n_paths

A = ReconstructSequence()  # Use the default file input_path

records = A.parse_with_position_column()

total_rows = len(records)
x_counter = 0
label_counter = 0
fold_changes = []

with open(A.output_path, 'w', newline='', encoding='utf-8') as ofh:
    writer = csv.writer(ofh)
    writer.writerow(['RefID', 'IsolateID', 'sequence', 'log2fold', 'label'])

    for i in range(total_rows):
        modified_sequence = A.ref_integrase
        print(f"\nrow:{i} {records[i]['metadata']['DTG']}", end='\t')
        for key, value in records[i]['positions'].items():
            if value != '-':
                new_aa = value[0]
                start_index = (key - 1) * 3
                end_index = start_index + 3
                old_codon = modified_sequence[start_index: start_index + 3]
                old_aa = A.genetic_code[old_codon]
                try:
                    new_codon = A.most_parsimonious_codon(
                        target_aa=new_aa, start_codon=modified_sequence[start_index: start_index + 3])[0]
                except ValueError:  # Occurs when the target AA is not in the genetic code - assume X ambiguity code
                    new_codon = old_codon
                    x_counter += 1
                print(f'{old_aa}{str(key)}{new_aa} {old_codon}->{new_codon}', end='\t')
                modified_sequence = modified_sequence[:start_index] + new_codon + modified_sequence[end_index:]
        print(f'{len(modified_sequence)}bp seq differs from ref: {modified_sequence != A.ref_integrase}')
        print(f'\n{modified_sequence}')
        ref_id = records[i]['metadata']['RefID']
        isolate_id = records[i]['metadata']['IsolateID']
        log2fold = log2(float(records[i]['metadata']['DTG']))  #  Log2 transform of fold change
        fold_changes.append(log2fold)
        label = 0 if log2fold < A.threshold else 1
        label_counter = label_counter + 1 if label == 0 else label_counter
        writer.writerow([ref_id, isolate_id, modified_sequence, log2fold, label])
print(f'\nNumber of Xs: {x_counter}')
print(f'Number of 0 labels: {label_counter} at threshold {A.threshold} out of {i} total rows ({(label_counter/i)*100:.2f}%)')
print('\n')

data = np.asarray(fold_changes)

# ------------------------------------------------------------------------------
#  Now plot the distribution of fold changes as histogram, gaussian, and KDE fit
# ------------------------------------------------------------------------------

# Robust histogram range: trim extreme tails
low, high = np.percentile(data, q=[0.0001, 99.9999])

# Create a two-panel figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# ---- Panel 1: Histogram + fits ----
ax = axes[0]

ax.hist(
    data,
    bins=30,
    range=(low, high),
    density=True,
    alpha=0.6,
    label="Histogram"
)

# X grid for fits
x = np.linspace(low, high, num=500)

# Normal fit
mu, sigma = norm.fit(data)
ax.plot(x, norm.pdf(x, mu, sigma), label="Normal")

# KDE fit
kde = gaussian_kde(data)
ax.plot(x, kde(x), label="KDE")

ax.set_xlabel("Log Fold Change")
ax.set_ylabel("Density")
ax.set_title("Distribution of Log2-Transformed Fold Changes")
ax.legend()

# ---- Panel 2: Q–Q plot ----
ax = axes[1]
probplot(data, dist="norm", plot=ax)
ax.set_title("Normal Q–Q Plot")

plt.tight_layout()
plt.show()


