import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help= "Input file")
    args = parser.parse_args()

    end1 = 322
    end2 = 644
    end3 = 966

    name1 = f"../data/Harmanpreet.tsv"
    name2 = f"../data/Yan.tsv"
    name3 = f"../data/Tristan.tsv"

    result1 = pd.DataFrame(columns = ["Name", "title", "coding"])
    result2 = pd.DataFrame(columns = ["Name", "title", "coding"])
    result3 = pd.DataFrame(columns = ["Name", "title", "coding"])

    j1 = j2 = j3 = 0

    data = pd.read_csv(args.input_file, sep="\t")

    for i in range(967):
        if i <= end1:
            result1.loc[j1] = [data.loc[i, "Name"], data.loc[i, "title"], data.loc[i, "coding"]]
            j1 = j1 + 1
        elif i <= end2:
            result2.loc[j2] = [data.loc[i, "Name"], data.loc[i, "title"], data.loc[i, "coding"]]
            j2 = j2 + 1
        elif i <= end3:
            result3.loc[j3] = [data.loc[i, "Name"], data.loc[i, "title"], data.loc[i, "coding"]]
            j3 = j3 + 1

    result1.to_csv(name1, index=False, sep="\t")
    result2.to_csv(name2, index=False, sep="\t")
    result3.to_csv(name3, index=False, sep="\t")

if __name__ == '__main__':
    main()