#/bin/bash
## Submit with `qsub -A <account>`
#PBS -N mosa2
#PBS -q casper
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=20:mem=80gb
#PBS -j oe

cd /glade/u/home/zmoon/git/TAMS

# Load conda env using NCAR conda installation
module load conda/latest
conda activate /glade/u/home/zmoon/mambaforge/envs/tams-dev

python run-mosa-step1.py
