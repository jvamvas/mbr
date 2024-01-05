import sys
import urllib
from collections import namedtuple
from pathlib import Path
from typing import Union, List

import jsonlines
import sacrebleu
import torch
from datasets import load_dataset
from fairseq import hub_utils
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.hub_utils import GeneratorHubInterface
from tqdm import tqdm

language_pair = sys.argv[1]
assert language_pair in ["de-en", "en-de", "en-ru", "ru-en"]

num_samples = 256
seed = 42
epsilon_cutoff = 0.02  # Different value might be needed because of label smoothing


class FairseqTranslationModel:
    """
    Copied from https://github.com/ZurichNLP/contrastive-conditioning/blob/master/translation_models/fairseq_models.py
    """

    def __init__(self,
                 name: str,
                 model: GeneratorHubInterface = None,
                 model_name_or_path: Union[Path, str] = None,
                 checkpoint_file: str = "checkpoint_best.pt",
                 src_bpe_codes: Union[Path, str] = None,
                 tgt_bpe_codes: Union[Path, str] = None,
                 max_tokens: int = 1000,
                 **kwargs,
                 ):
        self.name = name
        self.model = model or hub_utils.GeneratorHubInterface(**hub_utils.from_pretrained(
            model_name_or_path=str(model_name_or_path),
            checkpoint_file=checkpoint_file,
            **kwargs,
        ))
        # self.model.args.max_tokens = max_tokens
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        # EN-RU systems use separate vocabularies, which is not yet supported by torch hub
        bpe_args = namedtuple("bpe_args", ["bpe_codes"])
        if src_bpe_codes is not None:
            bpe_args_src = bpe_args(bpe_codes=str(src_bpe_codes))
            self.src_bpe = fastBPE(bpe_args_src)
        else:
            self.src_bpe = None
        if tgt_bpe_codes is not None:
            bpe_args_tgt = bpe_args(bpe_codes=str(tgt_bpe_codes))
            self.tgt_bpe = fastBPE(bpe_args_tgt)
        else:
            self.tgt_bpe = None

    def translate(self, sentences: List[str], beam: int = 5, **kwargs) -> List[str]:
        return self.model.translate(sentences, beam, **kwargs)

    def sample(self, sentences: List[str], **kwargs) -> List[str]:
        return self.model.sample(sentences, sampling=True, **kwargs)

    def __str__(self):
        return self.name


def load_model(language_pair: str) -> FairseqTranslationModel:
    if language_pair in ["de-en", "en-de"]:
        hub_interface = torch.hub.load(
            repo_or_dir="jvamvas/fairseq:epsilon",
            model=f'transformer.wmt19.{language_pair}.single_model',
            # checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt",
            tokenizer='moses',
            bpe='fastbpe',
        )
        model_name = f"transformer.wmt19.{language_pair}.single_model"
        model = FairseqTranslationModel(
            name=model_name,
            model=hub_interface,
        )
    elif language_pair in ["en-ru", "ru-en"]:
        hub_interface = torch.hub.load(
            repo_or_dir="jvamvas/fairseq:epsilon",
            model=f'transformer.wmt19.{language_pair}.single_model',
            # checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt",
            tokenizer='moses',
            bpe='fastbpe',
        )

        # Need to download correct vocab separately (https://github.com/pytorch/fairseq/issues/2928)
        hub_base_dir = Path(torch.hub.get_dir())
        correct_en_vocab_path = hub_base_dir / "en24k.fastbpe.code"
        correct_ru_vocab_path = hub_base_dir / "ru24k.fastbpe.code"
        if not correct_en_vocab_path.exists():
            with urllib.request.urlopen("https://dl.fbaipublicfiles.com/fairseq/en24k.fastbpe.code") as response, \
                    open(correct_en_vocab_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
        if not correct_ru_vocab_path.exists():
            with urllib.request.urlopen("https://dl.fbaipublicfiles.com/fairseq/ru24k.fastbpe.code") as response, \
                    open(correct_ru_vocab_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)

        evaluator_name = f"transformer.wmt19.{language_pair}.single_model"
        model = FairseqTranslationModel(
            name=evaluator_name,
            model=hub_interface,
            src_bpe_codes=correct_en_vocab_path if language_pair == "en-ru" else correct_ru_vocab_path,
            tgt_bpe_codes=correct_ru_vocab_path if language_pair == "en-ru" else correct_en_vocab_path,
        )
    else:
        raise NotImplementedError
    return model


src_path = sacrebleu.get_source_file("wmt19", language_pair)
dataset = load_dataset("text", data_files={"test": src_path})
source_sentences = dataset["test"]["text"]

model = load_model(language_pair)

out_dir = Path(__file__).parent / f"samples"
out_dir.mkdir(exist_ok=True, parents=True)
out_path = out_dir / f"{model}.{num_samples}samples.epsilon{epsilon_cutoff}.seed{seed}.jsonl"

with jsonlines.open(out_path, "w") as f:
    for source_sentence in tqdm(source_sentences):
        f.write({
            "samples": model.sample(num_samples * [source_sentence], sampling_epsilon_cutoff=epsilon_cutoff)
        })
