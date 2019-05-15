#!/bin/bash

# convert the avi video to images
#   Usage (sudo for the remove priviledge):
#       sudo ./convert_video_to_images.sh path/to/video fps
#   Example Usage:
#       sudo ./convert_video_to_images.sh ~/document/videofile/ 5
#   Example Output:
#       ~/document/videofile/walk/video1.avi 
#       #=>
#       ~/document/videofile/walk/video1/00001.jpg
#       ~/document/videofile/walk/video1/00002.jpg
#       ~/document/videofile/walk/video1/00003.jpg
#       ~/document/videofile/walk/video1/00004.jpg
#       ~/document/videofile/walk/video1/00005.jpg
#       ...

# In our case: 
#   sudo ./convert_video_to_images.sh ~/ucfcrimes/Videos 30

#SBATCH --partition=tibet --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:4

#SBATCH --job-name="sample"
#SBATCH --output=ffmpeg-%j.out

# only use the following if you want email notification
#SBATCH --mail-user=brianlui@stanford.edu
#SBATCH --mail-type=ALL

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

for folder in $1/*
do
    for file in "$folder"/*.mp4
    do
        if [ -z "$(ls -A ${file[@]%.mp4})" ]; then
            ffmpeg -f mp4 -i "$file" -vf fps=$2 "${file[@]%.mp4}"/%05d.jpg
        fi
    done
done
