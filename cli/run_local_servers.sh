# !/usr/bin/env bash

#################
# Parse options #
#################

instructions() {
  echo "Usage: $0 [-n] [-c]" >&2
  echo " -n: number of servers to run" >&2
  echo " -c: path to the server configs" >&2
  exit 1
}

if [ $# != 4 ]; then
    instructions
fi

while getopts ":n:c:t:" option; do
    case $option in
        n)  NUM_SERVERS=${OPTARG}
            ;;
        c)  CONFIG_PATH=${OPTARG}
            ;;
        \?) instructions
            ;;
   esac
done


###########################
# Install or activate env #
###########################

source ~/miniconda3/etc/profile.d/conda.sh
if conda env list | grep ".*bloom-demo.*"  >/dev/null 2>/dev/null; then
    conda activate bloom-demo
else
    conda create -y --name bloom-demo python=3.8.12 pip
    conda activate bloom-demo

    conda install -y -c conda-forge cudatoolkit-dev==11.3.1 cudatoolkit==11.3.1 cudnn==8.2.1.32
    pip install -i https://pypi.org/simple torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -i https://pypi.org/simple -r requirements.txt
fi


#######################
# Create Initial peer #
#######################

hivemind-dht &> tmp.out &
sleep 5
INITIAL_PEER=$(python -c "with open('tmp.out') as f: print(f.readlines()[1].split()[-1])" )
echo "Initial peer: ${INITIAL_PEER}"


##############################
# Initialize the config file #
##############################

typeset -A cfg 
cfg=( # set default values in config array
    [device]="cpu"
    [block_ids]="1:2"
    [id_path]="server.id"
    [maddr]="/ip4/127.0.0.1/tcp/30000"
)

###############
# Run servers #
###############

for SERVER_ID in $(seq 0 $(( $NUM_SERVERS - 1 )) )
do  
    ###############
    # Read config #
    ###############

    while read line
    do
        if echo $line | grep -F = &>/dev/null
        then
            varname=$(echo "$line" | cut -d '=' -f 1)
            cfg[$varname]=$(echo "$line" | cut -d '=' -f 2-)
        fi
    done < ${CONFIG_PATH}/server_${SERVER_ID}.cfg
    
    echo "=== Server #${SERVER_ID} ==="
    echo "Server ID: ${cfg[id_path]}"
    echo "Device: ${cfg[device]}"
    echo "Bloom block ids: ${cfg[block_ids]}"
    echo "Host maddr: ${cfg[maddr]}"
    echo ""
    
    ##############
    # Run server #
    ##############

    tmux new-session -d -s "Server_${SERVER_ID}" bash cli/deploy_server.sh -m "bigscience/test-bloomd" -i ${INITIAL_PEER} -d ${cfg[device]} -p ${cfg[id_path]} -b ${cfg[block_ids]} -a ${cfg[maddr]}
done

#####################
# Kill initial peer #
#####################

sleep 10
pkill -f hivemind-dht # TODO: kill only particular pids of hivemind-dht
rm tmp.out