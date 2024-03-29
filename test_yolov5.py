    
import torch 
from yolov5.yolo_infer import load_yolov5_model, infer_video
from pathlib import Path


weight = "base_model_state_dict.pt"
device = torch.device(0)
model = load_yolov5_model(weight, device)
video_file = "145307.mp4"
conf, iou = 0.45, 0.25
vid_stride, frame_per_sec = 25, 25 # 每秒推理1次
save_root = Path("runs") / Path(video_file).stem
classes = [7] # 人是第7类
imgsz = (640,640)
infer_video(model, video_file, conf, iou, device, classes, vid_stride, save_root, frame_per_sec, imgsz)