import os
import sys
from pathlib import Path
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5_models.common import DetectMultiBackend
from yolov5_utils.dataloaders import LoadImages
from yolov5_utils.general import check_img_size, cv2, non_max_suppression, scale_boxes
from yolov5_utils.torch_utils import smart_inference_mode
from dataclasses import dataclass
import numpy as np



def load_yolov5_model(weight, device):
    model = DetectMultiBackend(weight, device)
    # model.warmup()
    return model

@dataclass
class DetOut:
    xyxy: np.ndarray
    class_id: int
    conf: float

def image2input(img, device):
    # Padded resize
    # img = letterbox(img0, imgsz, stride=stride)[0]
    # # Convert
    # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    # img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # expand for batch dim
    return img

@smart_inference_mode()
def yolov5_det_infer(model, img, raw_img, conf_thres, iou_thres, device, classes, imgsz=(640,640), **kwargs):
    
    raw_shape = raw_img.shape[:2] 
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride) 
    input = image2input(img, device)
    
    pred = model(input, **kwargs)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes)
    
    res = []

    for i, det in enumerate(pred):  # per image
        if len(det):
            det[:, :4] = scale_boxes(input.shape[2:], det[:, :4], raw_shape).round()
            
            for *xyxy, conf, cls in reversed(det.cpu().numpy()):
                res.append(DetOut(
                    np.array(xyxy, dtype=np.int64), int(cls), float(conf)
                ))
    return res


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)


def get_time(i, vid_stride, frame_per_sec):
    total_sec = (i*vid_stride) // frame_per_sec
    hours, minutes, seconds = seconds_to_hms(total_sec)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def infer_video(model, video_file, conf_thres, iou_thres, device, classes, vid_stride, save_root, frame_per_sec=25, imgsz=(640,640)):
    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True, parents=True)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    dataset = LoadImages(path=video_file, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    
    for index, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        outputs = yolov5_det_infer(model, im, im0s, conf_thres, iou_thres, device, classes, imgsz)
        
        curr_frame_time = get_time(index, vid_stride, frame_per_sec) 
        for i, out in enumerate(outputs):
            xyxy = out.xyxy
            crop_img = im0s[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2], :]
            save_name = save_root / f"{Path(video_file).name}_{curr_frame_time}_{i:03}.jpg"
            cv2.imwrite(str(save_name), crop_img)




if __name__ == '__main__':
    # opt = parse_opt()
    # main(opt)
    weight = "/home/zhangqin/hesenxu/llm/base_model.pt"
    device = torch.device(0)
    model = load_yolov5_model(weight, device)
    video_file = "/home/zhangqin/hesenxu/llm/145307.mp4"
    conf, iou = 0.45, 0.25
    vid_stride, frame_per_sec = 25, 25 # 每秒推理1次
    save_root = Path("runs") / Path(video_file).stem
    classes = [7] # 人是第7类
    imgsz = (640,640)
    infer_video(model, video_file, conf, iou, device, classes, vid_stride, save_root, frame_per_sec, imgsz)