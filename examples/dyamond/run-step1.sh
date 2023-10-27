#/bin/bash
## Submit with `qsub -A <account>`
#PBS -N dyamond1
#PBS -q casper
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=20:mem=80gb
#PBS -j oe

cd /glade/u/home/zmoon/git/TAMS/examples/dyamond

py=/glade/u/home/zmoon/mambaforge/envs/tams-dev/bin/python

$py run-step1.py
