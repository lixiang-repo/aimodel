start_date=202312312359
end_date=203012012359
delta=24
infer_delta=192
time_format=%Y%m%d/part*
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
task=$(basename ${code_dir})
model_dir="/data/lixiang/model/${task}"
nas_path="/data/lixiang/data/${task}"
############################################################################################################
############################################################################################################
donefile=${model_dir}/donefile

mkdir -p ${model_dir} > /dev/null
mkdir -p ${model_dir}/logs > /dev/null

touch ${donefile}
