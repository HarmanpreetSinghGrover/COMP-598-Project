import json
import pandas as pd
import re
import argparse

"""
    This script takes a json file having one post per line, converts it into json.
    OUTPUT :
        saved in : ../data/post_dates.json
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
#     result = pd.DataFrame(columns = ["Name", "title", "coding", "date"])
    result = {}
#     j = 0
    cnt = 1
#     coding_ = ""
    for line in open(args.input_file, 'r'):
        post = json.loads(line)
        title_ = post["title"]
        if contains_trump_biden(title_):
            name_ = post["name"]
            if cnt <= 800:
                date_ = '21/Nov/2020'
            elif cnt <=1400:
                date_ = '22/Nov/2020'
            else:
                date_ = '23/Nov/2020'
#             result.loc[j] = [name_, title_, coding_, date_]
#             j = j + 1
            result[name_] = date_
        cnt += 1

    OUTPUT_FILE = f"../data/post_dates.json"
#     result.to_csv(OUTPUT_FILE, index=False, sep="\t")
    with open(OUTPUT_FILE, "w") as outfile:
        json.dump(result,outfile)

if __name__ == '__main__':
    main()