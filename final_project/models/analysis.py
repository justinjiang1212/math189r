import pandas as pd 

trump = pd.read_csv("/Users/andyliu/Downloads/trump_output.csv")
biden = pd.read_csv("/Users/andyliu/Downloads/biden_output.csv")

diffs = {}

for index in biden['Topic']:
    try:
        bval = biden.loc[index, 'Compound']
        tval = trump.loc[index, 'Compound']
        diff = float(bval) - float(tval)
        diffs[index] = diff
    except KeyError:
        pass

diff = {k: v for k, v in sorted(diffs.items(), key=lambda item: item[1])}
print(diff)