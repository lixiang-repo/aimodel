start_date=202312312359
end_date=203012012359
delta=24
time_format=%Y%m%d/part*
task=nasi_ctr_v1
model_dir="/data/lixiang/model/${task}"
nas_path="/data/lixiang/data/${task}"

############################################################################################################
############################################################################################################
donefile=${model_dir}/donefile

mkdir -p ${model_dir} > /dev/null
mkdir -p ${model_dir}/logs > /dev/null

touch ${donefile}
