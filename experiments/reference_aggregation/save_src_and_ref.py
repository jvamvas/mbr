import argparse
from pathlib import Path

from experiments.reference_aggregation.experiment_utils import Testset


parser = argparse.ArgumentParser()

parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
parser.add_argument('--limit-segments', type=int, default=None, help='Limit number of segments that are processed (used for testing)')

args = parser.parse_args()

translations_dir = Path(__file__).parent / "translations"
translations_dir.mkdir(exist_ok=True)

testset = Testset.from_wmt(args.testset, args.language_pair, limit_segments=args.segments)

src_out_path = translations_dir / f"{args.testset}.{args.language_pair}.src.{args.language_pair.split('-')[0]}"

with open(src_out_path, "w") as f:
    for src in testset.source_sentences:
        f.write(src + "\n")

ref_out_path = translations_dir / f"{args.testset}.{args.language_pair}.ref.{args.language_pair.split('-')[1]}"
with open(ref_out_path, "w") as f:
    for ref in testset.reference_sentences:
        f.write(ref + "\n")

print(f"Saved {len(testset.source_sentences)} source segments to {src_out_path}")
print(f"Saved {len(testset.reference_sentences)} reference segments to {ref_out_path}")
