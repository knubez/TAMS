#/bin/bash
## Submit with `qsub -A <account>`
#PBS -N dyamond2
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=21:mem=84gb
#PBS -j oe

cd /glade/u/home/zmoon/git/TAMS/examples/dyamond

py=/glade/u/home/zmoon/mambaforge/envs/tams-dev/bin/python

$py run-step2.py
