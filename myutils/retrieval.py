
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from myutils.retrieval_dataset import RetrievalImageDataset, RetrievalTextDataset
from myutils.retrieval_metric import retrieval_metric
from myutils.model_infer import extract_img_feats, extract_text_feats

import torch

def evaluate(model, tokenizer, img_root, test_text_file, batch_size, device, top_k, save_img_feats=True):
    img_root = Path(img_root)
    save_img_feats_filename = img_root / "img_feats.npy"
    save_img_files_filename = img_root / "img_files.txt"

    if save_img_feats_filename.exists() and save_img_files_filename.exists():
        img_feats = np.load(save_img_feats_filename)
        img_feats = torch.from_numpy(img_feats).to(device)
        img_files = np.loadtxt(save_img_files_filename, dtype=str)
    else:
        img_dataset = RetrievalImageDataset(img_root)
        img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        img_feats, img_files = extract_img_feats(model, device, img_dataloader)
        if save_img_feats:
            np.savetxt(save_img_files_filename, img_files, fmt="%s")
            np.save(save_img_feats_filename, img_feats.cpu().numpy())

    text_dataset = RetrievalTextDataset(test_text_file)
    text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    text_feats, image_files_texts = extract_text_feats(model, tokenizer, device, text_loader)
    sims_matrix = img_feats @ text_feats.t()
    sims_matrix = sims_matrix.t().cpu().numpy()
    sort_res = np.argsort(sims_matrix)[:, ::-1]

    pre = img_files[sort_res]
    ground_thruth = image_files_texts[:, 0]
    
    # hit ratio, mean reciprocal rank
    hr, mrr = retrieval_metric(ground_thruth, pre, top_k) 
    return hr, mrr


def retrieval_person_based_on_text(model, tokenizer, device, text, img_root, batch_size, save_img_feats=True):
    if isinstance(text, str):
        text = [text]
    text = [{"caption": i} for i in text]
    img_root = Path(img_root)
    save_img_feats_filename = img_root / "img_feats.npy"
    save_img_files_filename = img_root / "img_files.txt"

    if save_img_feats_filename.exists() and save_img_files_filename.exists():
        img_feats = np.load(save_img_feats_filename)
        img_feats = torch.from_numpy(img_feats).to(device)
        img_files = np.loadtxt(save_img_files_filename, dtype=str)
    else:
        img_dataset = RetrievalImageDataset(img_root)
        img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        img_feats, img_files = extract_img_feats(model, device, img_dataloader)
        if save_img_feats:
            np.savetxt(save_img_files_filename, img_files, fmt="%s")
            np.save(save_img_feats_filename, img_feats.cpu().numpy())
    
    text_dataset = RetrievalTextDataset(text)
    text_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    text_feats, _ = extract_text_feats(model, tokenizer, device, text_dataloader)

    sims_matrix = img_feats @ text_feats.t()
    sims_matrix = sims_matrix.t().cpu().numpy()
    sort_res = np.argsort(sims_matrix)[:, ::-1]

    pre_rank = img_files[sort_res]
    
    return pre_rank, np.sort(sims_matrix)[:, ::-1]


