import os
import json
import csv
import sys

import argparse
from tqdm import tqdm


def get_annotations(dataset):
    with open(dataset, 'r') as f:
        data = json.load(f)
    return data["annotations"]


def annotation_to_entry(annotation):
    return {"imageId": annotation["imageId"], "labelId": " ".join(map(str, annotation["labelId"]))}


def write_annotations_to_csv(annotations, csvfilename):

    fieldnames = ["imageId", "labelId"]
    with open(csvfilename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(map(annotation_to_entry, tqdm(annotations)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to the dataset to download.")
    parser.add_argument("csvfilename", help="Path to the the csvfile to write to.")
    args = parser.parse_args()

    # Catch the case where the file already exists
    if os.path.exists(args.csvfilename):
        response = ""
        answers = {"yes", "no"}
        while response not in answers:
            print("WARNING: %s already exists!" % args.csvfilename)
            response = input("Overwrite? (yes or no): ")
        if response == "no":
            sys.exit(1)

    annotations = get_annotations(args.dataset)
    write_annotations_to_csv(annotations, args.csvfilename)