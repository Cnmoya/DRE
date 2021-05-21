#!/bin/env python

import os
import argparse
from subprocess import run


def run_psfex(input_file, args):
    pass
    # command = ["sex"] + [input_file] + options
    # run(command)


if __name__ == "__main__":

    def parse_arguments():
        parser = argparse.ArgumentParser(description='Wrapper for running PSFex with DRE parameters')

        parser.add_argument('-i', "--input", help="directory with SExtractor catalogs", type=str, default="Sextracted")
        parser.add_argument('-o', "--output", help="output directory for the PSF's (def: PSF)",
                            type=str, default="PSF")
        parser.add_argument('-c', "--config", help="PSFex configuration file (def: default.psfex)",
                            type=str, default="default.psfex")

        args = parser.parse_args()

        return args

    args_ = parse_arguments()

    _, _, files = next(os.walk(args_.tiles))
    for file in sorted(files):
        run_psfex(f"{args_.input}/{file}", args_)
