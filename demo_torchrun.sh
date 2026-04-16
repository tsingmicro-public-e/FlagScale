timelog=`date "+%Y%m%d_%H%M%S"`

if [ $# -le 1 ]; then
        echo "[ERROR] Not Set Devices and rank ID!"
        exit
elif [ $# -eq 2 ]; then
        txda_ids=$1
        rank=$2
elif [ $# -eq 3 ]; then
        start_num=$1
        count=$2
        rank=$3
        txda_ids=""
        for ((i = 0; i < count; i++)); do
                if [ $i -eq $((count - 1)) ]; then
                        txda_ids="$txda_ids$((start_num + i))"
                else
                        txda_ids="$txda_ids$((start_num + i)),"
                fi
        done
fi

workdir=`pwd`
export PYTHONPATH=$workdir:$workdir/flagscale/train:$workdir/../Megatron-LM-FL:$PYTHONPATH
export TXDA_VISIBLE_DEVICES="$txda_ids"
echo $TXDA_VISIBLE_DEVICES

MASTER_ADDR=127.0.0.1
MASTER_PORT=8889

#export GLOO_SOCKET_IFNAME=bond0
#export NCCL_IB_DISABLE=1

export MG_PLATFORM="txda"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TXDA_LAUNCH_KERNEL_SYNC=1
export TXDA_SKIP_OPS="contiguous,cat,to.dtype"

export PRECISION_PRIORITY=1
#export DUMP_KERNEL_ARGS=1
#export TRITON_DUMP_PATH=/root/.triton/dump 
#export TRITON_ALWAYS_COMPILE=1
#export TRITON_QUICK_MODE=1
#export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1

## ---- for debug log ----
#export VS_DEBUG=1
#export TX_LOG_LEVEL=info
#export TX_LOG_PREFIX=1
#export TX_LAUNCH_LOG_LEVEL=info
#export PT_TXDA_LOG_LEVEL=FALLBACK
#export NCCL_DEBUG=TRACE
#export NCCL_DEBUG_FILE=$workdir/log/nccl_${timelog}_%h_%p.log
## ---- for debug log ----


nnodes=1
nproc=1
tp=1
pp=1
dp=$((($nnodes*$nproc)/($tp*$pp)))
mbs=4
gbs=$((1*$mbs*$dp))
block=28
seq_len=512
dtype=bf16
iters=10

data_path=/login_home/malin/datasets/llama3-datasets/wudao_llama3bpe_content_document
tokenizer_path=/login_home/malin/datasets/Qwen3-0.6B

config="${dtype}.dp${dp}tp${tp}pp${pp}.block${block}.gbs${gbs}.sq${seq_len}"
outdir="./log/outputs_qwen3_06b_block${block}"
load_dir="ckpts/init_ckpts"
mkdir -p ./log

if [[ ${dtype} == "bf16" ]]; then
    DTYPE_CONFIG="--bf16 --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32"
else
    DTYPE_CONFIG=""
fi

#--flag-gems-unused "_softmax,_softmax_backward_data,repeat_interleave.self_int,repeat_interleave.self_Tensor" \
#--sequence-parallel \
#--use-distributed-optimizer \
#--profile \
#--use-pytorch-profiler \
#--profile-step-start 5 \
#--profile-step-end 6 \
torchrun \
        --nproc_per_node ${nproc} \
        --nnodes ${nnodes} \
        --node_rank ${rank} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        flagscale/train/megatron/train_gpt.py \
        --num-workers 16 \
        --tensor-model-parallel-size $tp \
        --pipeline-model-parallel-size $pp \
        --context-parallel-size 1 \
        ${DTYPE_CONFIG} \
        --disable-bias-linear \
        --reset-position-ids \
        --reset-attention-mask \
        --qk-layernorm \
        --distributed-backend flagcx \
        --log-interval 1 \
        --tensorboard-log-interval 1 \
        --tensorboard-dir ${outdir}/tensorboard \
        --wandb-save-dir ${outdir}/wandb \
        --save-interval 10000 \
        --load ${load_dir} \
        --ckpt-format torch \
        --save ${outdir}/checkpoints \
        --transformer-impl local \
        --legacy-tokenizer \
        --enable-flag-gems \
        --num-layers ${block} \
        --hidden-size 1024 \
        --ffn-hidden-size 3072 \
        --kv-channels 128 \
        --group-query-attention \
        --num-attention-heads 16 \
        --num-query-groups 8 \
        --seq-length ${seq_len} \
        --max-position-embeddings 40960 \
        --norm-epsilon 1e-06 \
        --use-rotary-position-embeddings \
        --rotary-base 1000000 \
        --swiglu \
        --normalization RMSNorm \
        --init-method-std 6e-3 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --clip-grad 1.0 \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --no-position-embedding \
        --no-rope-fusion \
        --no-persist-layer-norm \
        --no-gradient-accumulation-fusion \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-swiglu-fusion \
        --no-bias-dropout-fusion \
        --seed 42 \
        --micro-batch-size ${mbs} \
        --global-batch-size ${gbs} \
        --eval-iters 0 \
        --train-iters ${iters} \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --lr 0.003 \
        --min-lr 0.0003 \
        --lr-warmup-fraction 0.1 \
        --lr-decay-style cosine \
        --data-path ${data_path} \
        --split 1 \
        --no-mmap-bin-files \
        --tokenizer-type QwenTokenizerFS \
        --tokenizer-path ${tokenizer_path} \
        --vocab-size 151851 \
        --make-vocab-size-divisible-by 64 2>&1 | stdbuf -o0 tee $workdir/log/stdout_$timelog.log.$config.qwen

#--train-samples 29297664 \
#--lr-warmup-samples 2048000 \
#--train-iters $iters \
#--lr-warmup-fraction 0.1 \

