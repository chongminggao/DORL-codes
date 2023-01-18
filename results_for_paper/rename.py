# -*- coding: utf-8 -*-
# @Time    : 2023/1/15 22:32
# @Author  : Chongming GAO
# @FileName: rename.py
import os
import re

import pandas as pd

from results_for_paper.visual_utils import walk_paths





def rename(dirpath, filenames, pattern="\[fivar", repl="[var"):
    for filename in filenames:
        if filename[0] == '.' or filename[0] == '_':  # ".DS_Store":
            continue
        filepath = os.path.join(dirpath, filename)

        filename_sub = re.sub(pattern, repl, filename)
        # res = pattern.match(filename)
        print(filename)
        print(filename_sub)

        os.rename(os.path.join(dirpath, filename), os.path.join(dirpath, filename_sub))
        # with open(filepath, "r") as file:
        #     lines = file.readlines()


def rename_dataset(datasets, patterns, repl="[var"):
    for dataset in datasets:
        dirpath = os.path.join("./results", dataset)
        filenames = walk_paths(dirpath)
        print(f"Loading data for {dataset}...")
        for pattern in patterns:
            rename(dirpath, filenames, pattern, repl)
    print("rename done!")

if __name__ == '__main__':

    patterns=["\[fivar", "\[fvar", "\[avar", "\[evar"]
    datasets = ["kuairec_2023 2022"]
    datasets = ["kuairec_2023", "kuairand_l8"]
    datasets = ["kuairec_l8"]

    patterns=["\[eivar"]
    datasets = ["kuairand_lab5"]
    patterns=["\[ezvar"]
    datasets = ["kuairec_ez1","kuairec_ez2"]


    rename_dataset(datasets, patterns)


