#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
task=$(basename ${code_dir})

ps -ef | grep ${task} |grep -v grep | awk '{print $2}' | xargs kill -9

sleep 1
echo "result"
ps -ef | grep ${task}
