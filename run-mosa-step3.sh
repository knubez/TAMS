#/bin/bash
## Submit with `qsub -A <account>`
#PBS -N mosa3
#PBS -q casper
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

cd /glade/u/home/zmoon/git/TAMS

# Load conda env using NCAR conda installation
module load conda/latest
conda activate /glade/u/home/zmoon/mambaforge/envs/tams-dev

python run-mosa-step3.py