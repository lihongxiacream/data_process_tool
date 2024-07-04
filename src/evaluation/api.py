from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import os
import json
from run_char_rm import CharacterRM
from gpt4_evaluate import GPT4Evaluation
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import copy
app = FastAPI()

# 定义文件存储路径
RESULTS_DIR = "results/"
EVALUATION_RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation.jsonl")

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 定义 Pydantic 模型
class EvalDataItem(BaseModel):
    id: int
    role: str
    novel_name: str
    context: str
    model_output: str
    prompt: Optional[str] = None

class InputData(BaseModel):
    eval_data: List[EvalDataItem]
    character_profiles: Dict[str, Any]
    metric_prompt: Optional[str] = None

#characterEval upload files
@app.post("/rm_file/")
async def upload_files(eval_data: UploadFile = File(...), character_profiles: UploadFile = File(...)):
    # 保存上传的文件到指定位置
    try:
        # 读取上传文件的内容
        eval_data = await eval_data.read()
        character_profiles = await character_profiles.read()

        # 将文件内容解析为 JSON 数据
        eval_data = json.loads(eval_data)
        character_profiles = json.loads(character_profiles)

    except json.JSONDecodeError:
        return {"error": "File content is not valid JSON"}

    # 处理数据
    obj=CharacterRM()
    result=obj.evaluate(eval_data, character_profiles)

    # 返回处理后数据
    return {"evaluated_data": result}

#Download files
@app.get("/download_file/")
async def download_file():
    # 返回处理后文件的路径
    file_path = EVALUATION_RESULTS_PATH

    # 检查文件路径是否存在
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=os.path.basename(file_path))
    else:
        raise HTTPException(status_code=404, detail="Processed file not found")

#CharacterEval upload json data
@app.post("/rm_data/")
async def process_items(input_data: InputData):
    try:
        eval_data=input_data.eval_data
        eval_data_dicts = [item.dict() for item in eval_data]
        character_profiles=input_data.character_profiles

    except json.JSONDecodeError:
        return {"error": "File content is not valid JSON"}

    # 处理数据
    obj = CharacterRM()
    result = obj.evaluate(eval_data_dicts, character_profiles)

    # 返回处理后数据
    return {"evaluated_data": result}

#GPT4Evaluate upload files
@app.post("/gpt_file/")
async def gptupload_files(eval_data: UploadFile = File(...), character_profiles: UploadFile = File(...),metric_prompt: Optional[str] = Form(None)):
    # 保存上传的文件到指定位置
    try:
        # 读取上传文件的内容
        eval_data = await eval_data.read()
        character_profiles = await character_profiles.read()

        # 将文件内容解析为 JSON 数据
        eval_data = json.loads(eval_data)
        character_profiles = json.loads(character_profiles)
        #eval_data = copy.deepcopy(eval_data[:2])
    except json.JSONDecodeError:
        return {"error": "File content is not valid JSON"}

    # 处理数据
    obj=GPT4Evaluation(eval_data, character_profiles,metric_prompt)
    result=obj.process_data()

    # 返回处理后数据
    return {"evaluated_data": result}

#GPT4Evaluate upload json data
@app.post("/gpt_data/")
async def gptprocess_items(input_data: InputData):
    try:
        eval_data=input_data.eval_data
        eval_data_dicts = [item.dict() for item in eval_data]
        character_profiles=input_data.character_profiles
        metric_prompt=input_data.metric_prompt

    except json.JSONDecodeError:
        return {"error": "File content is not valid JSON"}

    # 处理数据
    obj=GPT4Evaluation(eval_data_dicts, character_profiles,metric_prompt)
    result=obj.process_data()

    # 返回处理后数据
    return {"evaluated_data": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=30235)
