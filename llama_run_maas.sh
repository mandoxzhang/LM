host=""
all_process_num=0
first_ip=""

hostfile=$1

if [ -n "$2" ]
then
    nproc_per_node=$2
else
    nproc_per_node=8
fi

echo "runing file: $0"
echo "hostfile: $hostfile"
echo "nproc_per_node: $nproc_per_node"

sleep 1s
# export NCCL_IB_HCA=^mlx5_3:1
# export NCCL_SOCKET_IFNAME=bond0:10.11.1.2

while read line
do
    if [[ $line != \#* ]]
    then
        # local_num=${line: -1}
        host=$host$line","
        if [ $all_process_num -eq 0 ]; then
          first_ip=$line  
        fi
        #all_process_num=`expr $all_process_num + $local_num`
        all_process_num=`expr $all_process_num + 1`
        # echo $line
        # echo $local_num
    fi
done < $hostfile
# echo $host
host=${host%?}
# cnt=cnt*8
echo "all host $host"
echo "host num: $all_process_num"
echo "first ip: $first_ip"

sleep 1s

export NCCL_PROTOS=2
# export DS_ACCELERATOR=musa
export MUSA_KERNEL_TIMEOUT=3600000

OMP_NUM_THREADS=13 colossalai run --hostfile $hostfile \
    --nproc_per_node=$nproc_per_node \
    run_clm_colo_llama_maas.py \
    --master_addr $first_ip \
    --master_port 41240 \
    --device musa \
    --model_name_or_path ./llama2_config \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --output_dir ./test-clm \
    --cache_dir ./gpt2_ckpt/wikitext/wikitext-2-raw \
    --block_size 2048 \
    --num_train_epochs 100 \
    --checkpointing_steps 1500 \
    --save_dir ./checkpoint/llama2 \
    --shardinit \
    # --resume_from_checkpoint ./checkpoint/llama2 \

#--hostfile ./hostfile \
#--include $host \
# bash llama_run_maas.sh hostfile 2> error_log/$(date "+%Y-%m-%d_%H:%M:%S")

