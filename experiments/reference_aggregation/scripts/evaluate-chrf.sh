#!/bin/bash

cd translations

# Define the language pairs in an array
declare -a language_pairs=("en-de" "de-en" "en-ru" "ru-en")

# File to store the output
output_file="../results/chrf.log"

# Iterate over each language pair
for lp in "${language_pairs[@]}"
do
    echo $lp

    # Extract src and tgt from lp
    IFS='-' read -ra ADDR <<< "$lp"
    src=${ADDR[0]}
    tgt=${ADDR[1]}

    echo "baselines"
    sacrebleu wmt22.${lp}.ref.${tgt} -i wmt22.${lp}.beam4.${tgt} -m chrf -b >> $output_file
    sacrebleu wmt22.${lp}.ref.${tgt} -i wmt22.${lp}.epsilon0.02.seed0.${tgt} -m chrf -b >> $output_file

    echo "chrf"
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.pairwise.n1024.epsilon0.02.seed0.chrf.${tgt} -m chrf -b >> $output_file
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate.n1024.epsilon0.02.seed0.chrf.${tgt} -m chrf -b >> $output_file
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.chrf.${tgt} -m chrf -b >> $output_file

    echo "cometinho"
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.pairwise.n1024.epsilon0.02.seed0.cometinho.${tgt} -m chrf -b >> $output_file
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate.n1024.epsilon0.02.seed0.cometinho.${tgt} -m chrf -b >> $output_file
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.cometinho.${tgt} -m chrf -b >> $output_file

    echo "comet22"
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.pairwise.n1024.epsilon0.02.seed0.comet22.${tgt} -m chrf -b >> $output_file
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate.n1024.epsilon0.02.seed0.comet22.${tgt} -m chrf -b >> $output_file
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.comet22.${tgt} -m chrf -b >> $output_file

    echo "coarse-to-fine"
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.coarse_to_fine.top20.n1024.epsilon0.02.seed0.chrf-to-comet22.${tgt} -m chrf -b >> $output_file
    sacrebleu wmt22.${lp}.ref.${tgt} -i mbr.wmt22.${lp}.aggregate_to_fine.top20.n1024.epsilon0.02.seed0.chrf-to-comet22.${tgt} -m chrf -b >> $output_file

    echo
done
