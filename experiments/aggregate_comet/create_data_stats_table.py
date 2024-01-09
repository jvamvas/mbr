from pathlib import Path

import jsonlines

# \begin{tabularx}{\textwidth}{Xccc}
# \toprule
# & \# Segments & \# Samples per segment & \# Unique samples per segment (avg.) \\
# \midrule
# \textit{newstest21} & & & \\
# \textsc{en–de} & & & \\
# \textsc{de–en} & & & \\
# \textsc{en–ru} & & & \\
# \textsc{ru–en} & & & \\
# \addlinespace
# \textit{newstest22} & & & \\
# \textsc{en–de} & & & \\
# \textsc{de–en} & & & \\
# \textsc{en–ru} & & & \\
# \textsc{ru–en} & & & \\
# \bottomrule
# \end{tabularx}

header = "\\begin{tabularx}{\\textwidth}{Xccc}\n\\toprule\n"
header += "& \\# Segments & \\# Samples per segment & \\# Unique samples per segment (avg.) \\\\\n\\midrule\n"
footer = "\\bottomrule\n\\end{tabularx}"

body = ""
body += "\\textit{newstest21} & & & \\\\\n"
for lang_pair in ["en-de", "de-en", "en-ru", "ru-en"]:
    path = f"samples_{'wmt21'}/transformer.wmt19.{lang_pair}.single_model.1024samples.epsilon0.02.seed0.jsonl"
    path = Path(path)
    assert path.exists(), f"Path {path} does not exist"
    with jsonlines.open(path) as reader:
        data = list(reader)
    num_segments = len(data)
    num_samples = len(data[0]["samples"])
    avg_num_unique_samples = sum([len(set([sample["text"] for sample in segment["samples"]])) for segment in data]) / num_segments
    body += "\\textsc{" + lang_pair.replace('-', '–') + "} & " + str(num_segments) + " & " + str(num_samples) + " & " + "{:.2f}".format(avg_num_unique_samples) + " \\\\\n"
body += "\\addlinespace\n"
body += "\\textit{newstest22} & & & \\\\\n"
for lang_pair in ["en-de", "de-en", "en-ru", "ru-en"]:
    path = f"samples_{'wmt22'}/transformer.wmt19.{lang_pair}.single_model.1024samples.epsilon0.02.seed0.jsonl"
    path = Path(path)
    assert path.exists(), f"Path {path} does not exist"
    with jsonlines.open(path) as reader:
        data = list(reader)
    num_segments = len(data)
    num_samples = len(data[0]["samples"])
    avg_num_unique_samples = sum([len(set([sample["text"] for sample in segment["samples"]])) for segment in data]) / num_segments
    body += "\\textsc{" + lang_pair.replace('-', '–') + "} & " + str(num_segments) + " & " + str(num_samples) + " & " + "{:.2f}".format(avg_num_unique_samples) + " \\\\\n"

print(header + body + footer)
