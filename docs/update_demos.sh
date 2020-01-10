#! /bin/bash 
# script to regenerate the demo documentation .rst files. must be run from docs/ folder

file_list=$(find ../demos -name "*.ipynb")

for i in "${file_list[@]}"
do
   jupyter nbconvert --output-dir='./example_demos' $i --to rst
done
