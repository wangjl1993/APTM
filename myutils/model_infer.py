
import numpy as np
import torch
import torch.nn.functional as F



@torch.no_grad()
def extract_img_feats(model, device, dataloader):
    model.eval()
    model.to(device)

    image_embeds, image_feats = [], []
    image_files = []
    for data, file in dataloader:
        data = data.to(device) 
        image_embed, _ = model.get_vision_embeds(data)
        image_feat = model.vision_proj(image_embed[:, 0, :])
        image_feat = F.normalize(image_feat, dim=-1)
        
        image_embeds.append(image_embed)
        image_feats.append(image_feat)
        image_files += list(file)

    image_feats = torch.cat(image_feats, dim=0)
    
    
    return image_feats, np.array(image_files)


@torch.no_grad()
def extract_text_feats(model, tokenizer, device, dataloader):
    image_files_texts = []
    text_feats = []
    for file, text in dataloader:

        text_input = tokenizer(list(text), padding='max_length', truncation=True, max_length=56, return_tensors="pt")
        text_input.to(device)
        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.text_proj(text_embed[:, 0, :])
        text_feat = F.normalize(text_feat, dim=-1)

        text_feats.append(text_feat)
        image_files_texts.append(np.stack([file, text], 1))

    text_feats = torch.cat(text_feats, dim=0)
    image_files_texts = np.concatenate(image_files_texts)

    return text_feats, image_files_texts

