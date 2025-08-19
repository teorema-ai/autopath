
    # cluster info
    ```
        # basis capabilities:
        scontrol show nodes
        # current state
        sinfo -N -l
        # jobs
        squeue
    ```
    # start an interactive job:
    ```
        # alloc: GPU
        salloc --nodelist rhubarb -N 1 -n 1 --mem=8G --gpus=1
        # alloc: CPU
        salloc --nodelist rhubarb -N 1 -n 1 --mem=2G
        # job params: OPTIONAL
        env | grep SLURM_
        env | grep SLURM_JOBID
        # start interactive shell:
        srun --jobid=$SLURM_JOBID --pty bash

        # on rhubarb: BEGIN
        nvidia-smi
        htop
        conda activate pancan-gigapath
        # ...
        exit
        # on rhubarb: END

        scancel $SLURM_JOBID

    # GPU
    export CUDA_VISIBLE_DEVICES="3"
