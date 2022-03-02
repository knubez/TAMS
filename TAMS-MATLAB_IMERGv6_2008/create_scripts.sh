#!/bin/bash

# use time stamps file generated using times.py
for ymd in `cat 2008-August-September.txt`; do
  echo $ymd

  matlab_script_name=processing_${ymd}.m
  sed -e "s/YMD/$ymd/" \
      processing_template.m > $matlab_script_name

  job_script_name=run_matlabscript_${ymd}.sh
  sed -e "s/JOB_NAME/mcs$ymd/" \
      -e "s/MATLAB_SCRIPT_NAME/${matlab_script_name%.m}/" \
      run_matlabscript_template.sh > $job_script_name

  qsub $job_script_name

done



