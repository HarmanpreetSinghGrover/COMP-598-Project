import json
import pandas as pd
import re
import argparse

"""
    This script takes a json file having one post per line, converts it into tsv.
    OUTPUT :
        saved in : ../data/clean_data.tsv
        format : <name><tag><title><tag><coding>
"""

def contains_trump_biden(title):
    trump = (re.search(r"[^\w\d][Tt][rR][uU][mM][pP][^\w\d]", title) != None)
    trump = trump | (re.search(r"^[Tt][rR][uU][mM][pP][^\w\d]", title) != None)
    trump = trump | (re.search(r"[^\w\d][Tt][rR][uU][mM][pP]$", title) != None)
    biden = (re.search(r"[^\w\d][Bb][iI][dD][eE][nN][^\w\d]", title) != None)
    biden = biden | (re.search(r"^[Bb][iI][dD][eE][nN][^\w\d]", title) != None)
    biden = biden | (re.search(r"[^\w\d][Bb][iI][dD][eE][nN]$", title) != None)
    return trump | biden

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
        if contains_trump_biden(title_):
            name_ = post["name"]
            result.loc[j] = [name_, title_, coding_]
            j = j + 1

    OUTPUT_FILE = f"../data/clean_data.tsv"
    result.to_csv(OUTPUT_FILE, index=False, sep="\t")

if __name__ == '__main__':
    main()