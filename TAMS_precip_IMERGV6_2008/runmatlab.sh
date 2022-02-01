#!/bin/csh
#
# Sample Batch Script for a SGI Altix MPI job
#
# Submit this script using the command: qsub mpi.pbs
#
# Use the "qstat" command to check the status of a job.
#
# The following are embedded QSUB options. The syntax is #PBS (the # does
# _not_ denote that the lines are commented out so do not remove).
#
#
# Set maximum wallclock time (hh:mm:ss)
#PBS -l walltime=48:00:00
#
#PBS -A open
#PBS -l nodes=1:ppn=1
#PBS -l pmem=220gb 
#
# Queue name (see info about other queues in web documentation)
#PBS -V
#
# Charge job to project (recommended for users with multiple projects)
# [If project is invalid, a valid project will be automatically selected]
##PBS -A TG-ATM100040
#
# Job name (default = name of script file)
#PBS -N run_script18
#
# Filename for standard output (default = <job_name>.o<job_id>)
#PBS -o matlab_run2008_10ms.out
#
# Filename for standard error (default = <job_name>.e<job_id>)
#PBS -e matlab_run2008_10ms.err
#
# Send mail when the job begins and ends (optional)
#PBS -m be
#------------------------------
# End of embedded QSUB options

#set echo # echo commands before execution; use for debugging

# Go to the job scratch directory

##cd /storage/home/hlh189/scratch/WRF/WRFV3/run_test_flat/
##cd /storage/home/hlh189/work/Codes_OffshoreMax/
cd /gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/TAMS_precip_IMERGV6_2008/

# Insert where script finds source for modules needed  
##source /opt/aci/modulefiles/Core
module load matlab/R2016a
limit coredumpsize 0
limit stacksize unlimited

# Run the MPI program on all nodes/processors requested by the job
# (program reads from input.8.10 and writes to output.8.10)
matlab-bin -nodisplay -nosplash -r overlapping_235_with_precip_new_latest
