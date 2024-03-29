from myutils.retrieval import evaluate
import ruamel.yaml as yaml
from models.model_retrieval import APTM_Retrieval
from models.tokenization_bert import BertTokenizer
import torch
from time import time

if __name__ == "__main__":
    config = yaml.load(open("configs/Retrieval_cuhk.yaml",'r'), Loader=yaml.Loader)
    device = torch.device("cuda:0")
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = APTM_Retrieval(config=config)
    model.load_pretrained("ft_cuhk/checkpoint_best.pth", config, is_eval=True)
    model = model.to(device)


    img_root = "/home/zhangqin/hesenxu/llm/human_retrieve_text"
    text_file = "test_dataset2.json"
    batch_size = 64
    top_k = 10

    start_time = time()
    hr, mrr = evaluate(model, tokenizer, img_root, text_file, batch_size, device, top_k, False)
    end_time = time()
    print(end_time-start_time)
    print(hr, mrr)

