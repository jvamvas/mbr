import argparse
from pathlib import Path

from experiment_utils import Testset
from fairseq_utils import load_model


parser = argparse.ArgumentParser()

parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
parser.add_argument("--beam-size", type=int, default=4)
parser.add_argument('--limit-segments', type=int, default=None, help='Limit number of segments that are processed (used for testing)')

args = parser.parse_args()

testset = Testset.from_wmt(args.testset, args.language_pair, limit_segments=args.segments)

model = load_model(args.language_pair)

translations_dir = Path(__file__).parent / "translations"
translations_dir.mkdir(exist_ok=True)
out_path = translations_dir / f"{args.testset}.{args.language_pair}.beam{args.beam_size}.{args.language_pair.split('-')[1]}"

translations = model.translate(testset.source_sentences, beam_size=args.beam_size)
assert len(translations) == len(testset.source_sentences)

with open(out_path, "w") as f:
    for translation in translations:
        f.write(translation + "\n")
