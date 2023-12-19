#!/usr/bin/env bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=project2
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --output=output-project2.out
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20480

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################
#SBATCH --qos=users
#SBATCH --account=users
#SBATCH --mail-type=END
#SBATCH --mail-user=kkacar20@ku.edu.tr

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

#load modules
module load cuda/11.8.0
module load gcc/9.3.0

echo "Running Job...!"
echo "==============================================================================="
echo "Running compiled binary..."


#echo "Parallel version with 16 threads"
#export OMP_NUM_THREADS=16
./mcubes -c -o -p 2