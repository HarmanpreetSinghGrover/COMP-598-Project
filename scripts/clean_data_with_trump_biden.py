import json
import pandas as pd
import re
import argparse

"""
    This script takes a json file having one post per line, converts it into tsv.
    OUTPUT :
        saved in : ../data/clean_data_with_trump_biden.tsv
        format : <name><tag><title><tag><coding>
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help= "File to be clean(with path)!")
    args = parser.parse_args()
    result = pd.DataFrame(columns = ["Name", "title", "coding"])
    j = 0
    coding_ = ""
    for line in open(args.input_file, 'r'):
        post = json.loads(line)
        title_ = post["title"]
  
        name_ = post["name"]
        result.loc[j] = [name_, title_, coding_]
        j = j + 1

    OUTPUT_FILE = f"../data/clean_data_with_trump_biden.tsv"
    result.to_csv(OUTPUT_FILE, index=False, sep="\t")

if __name__ == '__main__':
    main()
