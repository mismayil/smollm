#=== Modify these variables if needed. ===#
# Path to shared high-performance scratch filesystem.
SCRATCH=/scratch

# Path to nanotron codebase.
NANOTRON_DIR=$HOME/nanotron

# Path to smollm codebase.
SMOLLM_DIR=$HOME/smollm

# Path of the base nanotron configuration file to use.
BASE_CONFIG=$SMOLLM_DIR/text/pretraining/ee628/base_config.yaml

# Paths of the complementary nanotron configuration files to use (to e.g.
# specify architecture, data or optimizer).
NEW_CONFIGS=(
	$SMOLLM_DIR/text/pretraining/ee628/arch_smollm2_135M.yaml
	$SMOLLM_DIR/text/pretraining/ee628/data_slimpajama.yaml
)


# Variables used in OVERRIDE_CONFIG_KEYS below. You may delete them if you want to change the overriden keys.
RUN_NAME=smollm2-135M-slimpajama
CKPT_PATH=$SCRATCH/users/$(whoami)/checkpoints/$RUN_NAME

# Specific keys to override in the unified configuration file
# with `key1/key2/.../keyn:value` format. See `unify_config.py` for more information.
OVERRIDE_CONFIG_KEYS=(
	general/run:$RUN_NAME
	optimizer/zero_stage:0
	checkpoints/checkpoints_path:$CKPT_PATH
	checkpoints/resume_checkpoint_path:$CKPT_PATH
	checkpoints/checkpoint_interval:10
)

# Path where the directories with logfiles will be stored.
LOG_ROOT=$SMOLLM_DIR/logs

#=== Prelude: Initial configuration and prints. ===#
echo Start time: $(date)
set -e

cd $NANOTRON_DIR
pip install -e .

# Handle general variables depending if you are on the single or multinode training case.
if [ -z ${WORLD_SIZE+x} ]; then
	echo Single-node setup detected!
	NGPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
	NNODES=1
	RANK=0

	echo Num GPUs: $NGPUS_PER_NODE

	TORCHRUN_ARGS=(
		--nnodes 1
		--nproc-per-node $NGPUS_PER_NODE
	)
else
	echo Multi-node setup detected!
	NGPUS_PER_NODE=$PET_NPROC_PER_NODE
	NNODES=$(( WORLD_SIZE / NGPUS_PER_NODE ))
	export LOCAL_WORLD_SIZE=$NGPUS_PER_NODE

	echo Local world size: $LOCAL_WORLD_SIZE
	echo Num nodes: $NNODES
	echo Num GPUs: $(($NGPUS_PER_NODE * $NNODES))
	echo Rank: $RANK

	TORCHRUN_ARGS=(
		--nnodes $NNODES
		--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
		--rdzv_backend c10d
		--nproc-per-node $NGPUS_PER_NODE
		--node-rank $RANK
	)
fi

export HF_HOME=$SCRATCH/hf_home
export OMP_NUM_THREADS=192

# Create the unified config file. 
TEMP_CONFIG=$SMOLLM_DIR/.temp.yaml
python $SMOLLM_DIR/text/pretraining/unify_config.py \
	--base-config $BASE_CONFIG \
	--new-config-paths ${NEW_CONFIGS[@]} \
	--out-path $TEMP_CONFIG \
	--nnodes $NNODES \
	--gpus-per-node $NGPUS_PER_NODE \
	--overrides ${OVERRIDE_CONFIG_KEYS[@]}

SCRIPT_ARGS=(
	--config-file $TEMP_CONFIG
)

LOG_DIR=$LOG_ROOT/run_${SUBMIT_TIME}_$JOB_UUID
mkdir -p $LOG_DIR
LOG_PREFIX=$LOG_DIR/rank$RANK

#=== Launch command. ===#
CMD="torchrun ${TORCHRUN_ARGS[@]} $NANOTRON_DIR/run_train.py ${SCRIPT_ARGS[@]}"
echo "CMD: $CMD"
$CMD 2> >(tee $LOG_PREFIX.err >&2) 1> >(tee $LOG_PREFIX.out)  # write stdout and stderr to logfiles.
EXIT_CODE=$?

#=== Cleanup. ===#
echo "END TIME: $(date)"
exit $EXIT_CODE
