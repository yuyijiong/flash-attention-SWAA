#设置MAX_JOBS为32以加快编译速度
export MAX_JOBS=32

cd ./flash-attention/
python setup.py install
cd ../flash-attention-vllm/
python setup.py install

#nohup bash install.sh >install.log 2>&1 &
