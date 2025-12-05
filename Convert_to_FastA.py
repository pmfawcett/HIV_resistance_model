with open('New_sequences_labeled_with_Ns_no_labels.csv', "r") as fin, open('Ns_no_label.fasta', "w") as fout:
    for i, line in enumerate(fin):
        seq = line.strip()
        if not seq:
            continue
        fout.write(f">seq_{i}\n{seq}\n")