---
language: 
- en 
- zh 
tags:
- translation
widget: 
- text: "我喜欢学习数据科学和机器学习。"
---

# liam168/trans-opus-mt-zh-en

## Model description

* source group: English 
* target group: Chinese 
* model: transformer
* source language(s): eng

## How to use

```python
>>> from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline
>>> mode_name = 'liam168/trans-opus-mt-zh-en'
>>> model = AutoModelWithLMHead.from_pretrained(mode_name)
>>> tokenizer = AutoTokenizer.from_pretrained(mode_name)
>>> translation = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer)
>>> translation('我喜欢学习数据科学和机器学习。', max_length=400)
    [{'translation_text': 'I like to study data science and machine learning.'}]
```

## Contact

liam168520@gmail.com
