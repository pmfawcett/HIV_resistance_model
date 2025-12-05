import csv

infile = 'New_sequences_labeled_with_Ns.csv'
outfile = 'New_sequences_labeled_with_Ns_filtered.csv'
N = 25
print(f'Throwing away sequences with greater than {N} Ns')

with open(infile) as fin, open(outfile, "w") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    header = next(reader)
    writer.writerow(header)
    i = count = 0
    for row in reader:
        i += 1
        # print(row)
        try:
            n_count = row[0].count("N")
            if n_count <= N:
                writer.writerow(row)
                count += 1
            else:
                print(f'Rejected row {i} which had {n_count} Ns')
        except IndexError:
            pass
    print(f'Wrote a total of {count} sequences to {outfile} from a total of {i} sequences. {i-count} were rejected')
