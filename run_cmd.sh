host=""
all_process_num=0
first_ip=""

hostfile=$1

if [ -n "$2" ]
then
    nproc_per_node=$2
else
    nproc_per_node=1
fi

echo "runing file: $0"
echo "hostfile: $hostfile"
echo "nproc_per_node: $nproc_per_node"

sleep 1s


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

# export NCCL_IB_HCA=^mlx5_3:1
# export NCCL_SOCKET_IFNAME=bond0:10.11.1.2
export NCCL_PROTOS=2

OMP_NUM_THREADS=13 colossalai run --hostfile $hostfile \
    --nproc_per_node=$nproc_per_node \
    run_cmd.py \
    --master_addr $first_ip \
    --master_port 41218 \
    --mthreads

#--hostfile ./hostfile \
#--include $host \
# bash llama_run.sh hostfile 2> error_log/$(date "+%Y-%m-%d_%H:%M:%S").txt

