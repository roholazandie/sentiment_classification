import pytreebank
import sys
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
        "--raw_dataset_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the files of stanfordSentimentTreebank",
    )

parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output dir. all *.tsv files will be here",
    )

args = parser.parse_args()


dataset = pytreebank.load_sst(args.raw_dataset_dir)
out_path = os.path.join(args.output_dir, '{}.txt')

# Store train, dev and test in separate files
for category in ['train', 'test', 'dev']:
    with open(out_path.format(category), 'w') as outfile:
        for item in dataset[category]:
            outfile.write("{}\t{}\n".format(item.to_labeled_lines()[0][0] + 1, item.to_labeled_lines()[0][1]))
# Print the length of the training set
print(len(dataset['train']))