import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
from translate_ch_en.translation import Translation_model
from myutils.retrieval import retrieval_person_based_on_text
# import ruamel.yaml as yaml
import yaml
from models.model_retrieval import APTM_Retrieval
from models.tokenization_bert import BertTokenizer
import torch
from yolov5.yolo_infer import load_yolov5_model, infer_video



# APTM person retrieval model
config = yaml.load(open("configs/Retrieval_cuhk.yaml",'r'), Loader=yaml.FullLoader)
device = torch.device("cuda:0")
tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
model = APTM_Retrieval(config=config)
model.load_pretrained("ft_cuhk/checkpoint_best.pth", config, is_eval=True)
model = model.to(device)

# translate Chinese to English model. 
translate_model = Translation_model()

# yolov5 det model, crop person_image first.
yolo_model = load_yolov5_model("base_model_state_dict.pt", device)
conf, iou = 0.45, 0.25
vid_stride, frame_per_sec = 25, 25 # 每秒推理1次
classes = [7] # 人是第7类
imgsz = (640,640)

def re_func(text, retrieval_path, top_k):
    retrieval_path = Path(retrieval_path)
    if retrieval_path.is_dir():
        img_root = retrieval_path
    else:
        img_root = retrieval_path.parent / retrieval_path.stem
        img_root.mkdir(exist_ok=True, parents=True)
        if not img_root.exists():
            infer_video(yolo_model, str(retrieval_path), conf, iou, device, classes, vid_stride, img_root, frame_per_sec, imgsz)

    text = translate_model(text)
    retri_img_files, retri_scores = retrieval_person_based_on_text(model, tokenizer, device, text, img_root, 64)
    retri_img_files = retri_img_files[:, :top_k]
    retri_scores = retri_scores[:, :top_k]
    retri_img = [Image.open(i).resize((128, 256)) for i in retri_img_files[0]]
    retri_img = np.concatenate(retri_img, axis=1)
    return str(retri_img_files), retri_img, str(retri_scores)


with gr.Blocks() as demo:
    gr.Markdown("<center><h2>HITO海拓找人助手</h2></center>") # 使用 Markdown 输出一句话
    with gr.Tab("读取文件目录演示"): # 新建一个 Tab
        with gr.Row():
            with gr.Column(scale=1):
                retrieval_path = gr.Textbox(label="文件/文件夹路径", value="145307.mp4")
                text = gr.Textbox(label="查询语句", lines=4, value='一个穿着黑色外套，黑色长裤，黑色鞋子的男性。')
                top_k = gr.Slider(5, 20, step=1,label='top_k')
                text_button = gr.Button("查询")
            with gr.Column(scale=2):
                img_file_output = gr.Textbox(label="查询结果")
                img_output = gr.Image(show_label=False)
                scores_output = gr.Textbox(show_label=False)
                
    text_button.click(re_func, inputs=[text, retrieval_path, top_k], outputs=[img_file_output, img_output, scores_output]) # 按钮绑定相应的槽函数


demo.launch(server_name="0.0.0.0", share=True, show_error=True, debug=True)
