#!/bin/bash
#PBS -l walltime=48:00:00
#
#PBS -l nodes=1:ppn=20:scivybridge
##PBS -l nodes=1:ppn=20:himem
#PBS -l pmem=10gb
#
#
##PBS -A jle7_a_g_sc_default   
#PBS -A open
#
##PBS -V
#
#PBS -N JOB_NAME
#
# Filename for standard output (default = <job_name>.o<job_id>)
##PBS -o matlab_run.out
#
# Filename for standard error (default = <job_name>.e<job_id>)
##PBS -e matlab_run.err
#
#PBS -oe
#
## Send mail when the job begins and ends (optional)
##PBS -m be
#------------------------------
# End of embedded QSUB options

#set echo # echo commands before execution; use for debugging

# Go to the job scratch directory

##cd /storage/home/hlh189/scratch/WRF/WRFV3/run_test_flat/
##cd /storage/home/hlh189/work/Codes_OffshoreMax/
cd /gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/TAMS_precip_IMERGV6_2008/

# Insert where script finds source for modules needed  
##source /opt/aci/modulefiles/Core
module purge
module load matlab #/R2016a
#limit coredumpsize 0
#limit stacksize unlimited

# Run the MPI program on all nodes/processors requested by the job
# (program reads from input.8.10 and writes to output.8.10)
#matlab-bin -nodisplay -nosplash -r MATLAB_SCRIPT_NAME
matlab -nodisplay -nosplash -r MATLAB_SCRIPT_NAME


