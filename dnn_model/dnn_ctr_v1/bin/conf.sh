#start_date=20231001
start_date=20230928
start_date=20240416
#start_date=20240101
backup=20230401
delta=24
infer_delta=24
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
task=$(basename ${code_dir})
model_dir="/data/share/model/${task}"
############################################################################################################
############################################################################################################
mkdir -p ${model_dir}/logs > /dev/null

conda activate env
