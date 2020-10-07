#!/bin/bash

grep -v '^#' imstats.new.dat | awk '{printf "\\verb|%s| %s %s\n", $1, $6, $7}' > temp.out
csv_latex -i temp.out -d space > raw.tex
rm temp.out

sed -i '1s/^/\\begin{table}[ht]\n\\centering\n\\begin{tabular}{ccc}\n\\toprule\nfilename \& mean \& sigma \\\\ \n\\midrule \n/' raw.tex
printf '\\bottomrule\n\\end{tabular}\n' >> raw.tex
# printf '\\caption{Raw data for detector linearity and full-well capacity}\n' >> raw.tex
printf '\\end{table}' >> raw.tex

csv_latex -i fit.dat -d , > fit.tex
sed -i '1s/^/\\begin{table}[ht]\n\\centering\n\\begin{tabular}{ccccc}\n\\toprule\ntemperature \& slope \& sigma slope \& intercept \& sigma intercept\\\\ \n\\midrule \n/' fit.tex
printf '\\bottomrule\n\\end{tabular}\n' >> fit.tex
printf '\\end{table}' >> fit.tex
