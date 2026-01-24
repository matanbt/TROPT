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
    "info_similarities",
    "stuffing_similarities",
    "raslite_similarities",
    "combi_similarities",
]

# Create a dictionary of functions to apply to specific columns
converters = {col: json.loads for col in json_cols}

pretty_names = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM",
    "sentence-transformers/all-mpnet-base-v2": "aMPNet",
    "Snowflake/snowflake-arctic-embed-m": "Arctic",
    "intfloat/e5-base-v2": "E5",
    "sentence-transformers/gtr-t5-base": "GTR-T5",
    "facebook/contriever": "Contriever",
    "facebook/contriever-msmarco": "Contriever-MS",
    "sentence-transformers/msmarco-roberta-base-ance-firstp": "ANCE",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": "mMPNet",
    "openai/text-embedding-3-small": "OAI-Small\\footnotemark[1]",
    "openai/text-embedding-3-large": "OAI-Large\\footnotemark[1]",
    "google/gemini-embedding-001": "Gem-Large\\footnotemark[1]",
}


data = pd.read_csv("combi_attack3.csv", converters=converters)

res = ""
tok = ""

res += "\\documentclass[a4paper, sigconf]{acmart}\n"
res += "\\begin{document}\n"
res += "\\begin{tabular}{l|cccc}\n"
res += "\\hline\n"
res += "& \\multicolumn{4}{c}{appeared@1} \\\\\n"
res += "\\textbf{Model} & \\texttt{info} Only & stuffing & \\texttt{RASLITE} & \\textbf{Combi.} \\\\ \\hline\n"

tok += "\\documentclass[a4paper, sigconf]{acmart}\n"
tok += "\\begin{document}\n"
tok += "\\begin{tabular}{l|cc|c}\n"
tok += "\\hline\n"
tok += "& \\multicolumn{2}{c|}{Tokens} & \\\\\n"
tok += "\\textbf{Model} & \\texttt{RASLITE} & \\textbf{Combi.} & \\textbf{Reduction} ($\\times$) \\\\ \\hline\n"

r_tokens = []
c_tokens = []

for model in pretty_names.keys():
    if len(data[data["model"] == model]):
        x = data[data["model"] == model].iloc[0]
    else:
        res += f"{pretty_names[model]} & & & & \\\\ \\hline\n"
        tok += f"{pretty_names[model]} & & \\\\ \\hline\n"
        continue
    trials = x["trials"]

    # RESULT TABLE
    best_sim = np.array(x["best_similarities"])
    perc = lambda sim: int((np.count_nonzero(np.array(sim) > best_sim) / trials) * 100)

    res += f"{pretty_names[x["model"]]} & "
    res += f"{perc(x["info_similarities"])}\\% & "
    res += f"{perc(x["stuffing_similarities"])}\\% & "
    r = perc(x["raslite_similarities"])
    res += "\\textbf{" if r == 100 else ""
    res += f"{perc(x["raslite_similarities"])}\\%"
    res += "} & " if r == 100 else " & "
    res += "\\textbf{" + f"{perc(x["combi_similarities"])}\\%" + "} "

    res += "\\\\ \\hline\n"

    # TOKEN TABLE
    avg = lambda arr: int(np.average([item[-1] for item in arr]))

    tok += f"{pretty_names[x["model"]]} & "
    ras = avg(x["raslite_tokens"])
    r_tokens.append(ras)
    com = avg(x["combi_tokens"])
    c_tokens.append(com)
    tok += f"{ras:,} & "
    tok += "\\textbf{" + f"{com:,}" + "} & "
    tok += f"{ras / com:.2f}"
    tok += "\\\\ \\hline\n"

res += "\\end{tabular}\n"
res += "\\end{document}\n"

tok += "\\hline \n"
tok += "\\textbf{\\textit{Average}} & "
tok += f"{int(np.average(r_tokens)):,} & "
tok += "\\textbf{" + f"{int(np.average(c_tokens)):,}" + "} & "
tok += f"{np.average(r_tokens) / np.average(c_tokens):.2f}"
tok += "\\\\ \\hline\n"

tok += "\\end{tabular}\n"
tok += "\\end{document}\n"

with open("tex/combi_appendix/results_table.tex", "w") as f:
    f.write(res)

with open("tex/combi_appendix/tokens_table.tex", "w") as f:
    f.write(tok)
