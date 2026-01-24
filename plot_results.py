import json

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Define which columns contain JSON
json_cols = [
    "raslite_losses",
    "raslite_tokens",
    "combi_losses",
    "combi_tokens",
    "best_similarities",
]

# Create a dictionary of functions to apply to specific columns
converters = {col: json.loads for col in json_cols}

data = pd.read_csv("combi_attack3.csv", converters=converters)


def find_starting_tokens(x):
    tokens = []
    for i in range(x["trials"]):
        combi_starting_sim = 0
        for loss in x["raslite_losses"][i]:
            if loss < x["combi_losses"][i][0]:
                break
            combi_starting_sim += 1
        if combi_starting_sim >= len(x["raslite_tokens"][i]):
            combi_starting_sim = len(x["raslite_tokens"][i]) - 1
        starting_tokens = x["raslite_tokens"][i][combi_starting_sim]
        tokens.append(x["raslite_tokens"][i][-1] - starting_tokens)
    return tokens


for x in data.iloc:
    print(x["model"])
    raslite_tokens = [item[-1] for item in x["raslite_tokens"]]
    raslite_sim_tokens = find_starting_tokens(x)
    combi_tokens = [item[-1] for item in x["combi_tokens"]]
    avg_raslite = np.average(raslite_tokens)
    avg_raslite_sim = np.average(raslite_sim_tokens)
    avg_combi = np.average(combi_tokens)
    print(f"{avg_raslite_sim} ({avg_raslite})")
    print(avg_combi)
    print(f"diff: {avg_raslite_sim / avg_combi:.2f} ({avg_raslite / avg_combi:.2f})")
    if x["model"] == "openai/text-embedding-3-small":
        for i in range(x["trials"]):
            combi_starting_sim = 0
            for loss in x["raslite_losses"][i]:
                if loss < x["combi_losses"][i][0]:
                    break
                combi_starting_sim += 1
            if combi_starting_sim >= len(x["raslite_tokens"][i]):
                combi_starting_sim = len(x["raslite_tokens"][i]) - 1
            combi_starting_sim = x["raslite_tokens"][i][combi_starting_sim]

            plt.plot(
                x["combi_tokens"][i],
                [-float(loss) for loss in x["combi_losses"][i]],
                label="Combi Passage",
            )
            plt.plot(
                x["raslite_tokens"][i],
                [-float(loss) for loss in x["raslite_losses"][i]],
                label="RASLITE Passage",
            )
            plt.hlines(
                y=x["best_similarities"][i],
                color="g",
                xmin=0,
                xmax=max(x["combi_tokens"][i][-1], x["raslite_tokens"][i][-1]),
                label="Best Passage",
            )
            plt.vlines(
                x=combi_starting_sim,
                color="m",
                ymin=0,
                ymax=1,
                label="Combi Starting Similarity",
            )
            # plt.vlines(
            #     x=square_start, ymin=0, ymax=1, color="r", linestyle="--", label="Start of Square Attack"
            # )
            plt.xlabel("Number of Tokens")
            plt.ylabel("Similarity")
            plt.grid(True)
            plt.legend()
            plt.show()
