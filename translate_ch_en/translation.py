
from typing import Any
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
import warnings
from pathlib import Path
FILE_PATH = Path(__file__).resolve().parent
warnings.filterwarnings('ignore')
import torch


class Translation_model:
    """Translate English to Chinese; Translate Chinese to English
    """
    TRANSLATION_MODEL = {
        "translation_zh_to_en": FILE_PATH / "trans-opus-mt-zh-en",
        "translation_en_to_zh": FILE_PATH / "trans-opus-mt-en-zh"
    }

    def __init__(self, method: str = "translation_zh_to_en", device: torch.device=torch.device("cpu")) -> None:
        self.method = method
        self.model_name = self.TRANSLATION_MODEL[method]
        self.device = device
        self.lan_model = self._load_model(model_name=self.model_name)

    def _load_model(self, model_name: str):
        model = AutoModelWithLMHead.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        translation = pipeline(self.method, model=model, tokenizer=tokenizer, device=self.device)
        return translation

    def __call__(self, txt: str, max_length: int = 400, verbose:bool = False) -> Any:
        res = self.lan_model(txt, max_length=max_length)[0]["translation_text"]
        res = self.postprocess(res)
        if verbose:
            print(f">>> Translate: {res}")
        return res

    def postprocess(self, text: str):
        if self.method == "translation_zh_to_en":
            text = text.lower().strip()
            text = text.replace(" a ", " ")
        return text 

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        "Text Detection to Labelme File", add_help=True)
    # Chinese/English 
    parser.add_argument("-e", "--eng", action='store_true', default=False,
                        help="Translate English to Chinese. Default: Chinese to Enligsh")
    # txt prompt
    parser.add_argument("-t", "--text", type=str,
                        required=True, help="text")
    
    args = parser.parse_args()
    if args.eng:
        model = Translation_model(method= "translation_en_to_zh")
    else:
        model = Translation_model()

    # model("我喜欢学习数据科学和机器学习。", verbose=True)
    model(args.text, verbose=True)

