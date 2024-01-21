start_date=202311302359
end_date=203012012359
delta=24
time_format=%Y/%m/%d/part*
model_dir="/data/lixiang/nasi_ctr_model_v1"
nas_path="/data/lixiang/rm_show_click_v2"

############################################################################################################
############################################################################################################
donefile=${model_dir}/donefile

mkdir -p ${model_dir} > /dev/null
mkdir -p ${model_dir}/logs > /dev/null

touch ${donefile}
