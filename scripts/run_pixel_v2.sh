#!/bin/bash
# pixel localisation (corrected TIC star lists, 500 bootstraps from random starts) and the dilution table
set -e
cd "$(dirname "$0")"
rm -f ../results/pixel_v2.done
python3 s05a_pixel.py "TOI-678" "HIP 31126" "TOI-5997" > ../results/log_s05a_v2.txt 2>&1
python3 s04a_dilution.py > ../results/log_s04a_v2.txt 2>&1
touch ../results/pixel_v2.done
