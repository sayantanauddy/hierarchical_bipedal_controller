#!/usr/bin/env bash
image_path="${HOME}/computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/report/src"
filename_svg=${1}
filename_eps=$(echo $filename_svg | cut -f 1 -d '.')'.eps'
inkscape ${image_path}/${filename_svg} -E  ${image_path}/${filename_eps} --export-ignore-filters
echo "Converted ${filename_svg} to ${filename_eps}"