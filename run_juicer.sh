#pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
conda activate juicer_env

#gradio
streamlit run demos/process_QA_data/app.py --server.port 30235
streamlit run app.py --server.port 30235

# only for installation from source
python tools/process_data.py --config configs/demo/process.yaml

# use command line tool
dj-process --config configs/demo/process.yaml
