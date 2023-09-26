host=""
all_process_num=0
first_ip=""

hostfile=$1

if [ -n "$2" ]
then
    nproc_per_node=$2
else
    nproc_per_node=
fi

echo "runing file: $0"
echo "hostfile: $hostfile"
echo "nproc_per_node: $nproc_per_node"

sleep 1s

all_ip=()
while read line
do
    if [[ $line != \#* ]]
    then
        # local_num=${line: -1}
        all_ip+=("$line")
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

# export NCCL_IB_HCA=^mlx5_3:1
# export NCCL_SOCKET_IFNAME=bond0:10.11.1.2
export NCCL_PROTOS=2

process_line() {
    local first_ip="$1"
    local hostfile="$2"

    echo "Processing ip1: $first_ip, $hostfile"
    
    OMP_NUM_THREADS=13 colossalai run --hostfile $hostfile \
        --include $first_ip \
        --nproc_per_node=1 \
        run_cmd_iteration.py \
        --master_addr $first_ip \
        --master_port 41246 \
        --musa

}

for line in "${all_ip[@]}"; do
    process_line "$line" "$hostfile"
done


# OMP_NUM_THREADS=13 colossalai run --hostfile $hostfile \
#     --nproc_per_node=1 \
#     run_cmd.py \
#     --master_addr $first_ip \
#     --master_port 41246 \
#     --device musa \
#     --model_name_or_path ./chatglm-6b \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --per_device_train_batch_size 3 \
#     --per_device_eval_batch_size 1 \
#     --output_dir ./test-clm \
#     --cache_dir ./gpt2_ckpt/wikitext/wikitext-2-raw \
#     --block_size 2048 \
#     --num_train_epochs 10 \
#     --shardinit
    

#--hostfile ./hostfile \
#--include $host \
# bash llama_run.sh hostfile 2> error_log/$(date "+%Y-%m-%d_%H:%M:%S").txt