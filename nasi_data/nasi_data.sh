
#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"
cd ${code_dir}
source ~/.bashrc
set +x && source /root/mambaforge-pypy3/etc/profile.d/conda.sh && conda activate py && set -x

date=`date "+%Y%m%d"`
tasks=(nasi_cvr_v1 nasi_ctr_v1)
for task in ${tasks[@]}
do
    jupyter nbconvert --to html --execute ${code_dir}/${task}.ipynb --output /data/lixiang/logs/${date}_${task}
done

