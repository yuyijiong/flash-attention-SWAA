export MAX_JOBS=4

cd ./flash-attention/
python setup.py install
cd ../flash-attention-vllm/
python setup.py install

#nohup bash install.sh >install.log 2>&1 &
