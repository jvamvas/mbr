import argparse
from pathlib import Path

from experiment_utils import Testset
from fairseq_utils import load_model


def main(testset: str, language_pair: str, beam_size: int = 4, limit_segments: int = None, out_dir: Path = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent

    testset = Testset.from_wmt(testset, language_pair, limit_segments=limit_segments)
    
    model = load_model(language_pair)
    
    translations_dir = out_dir / "translations"
    translations_dir.mkdir(exist_ok=True)
    out_path = translations_dir / f"{testset}.{language_pair}.beam{beam_size}.{language_pair.split('-')[1]}"
    
    translations = model.translate(testset.source_sentences, beam_size=beam_size)
    assert len(translations) == len(testset.source_sentences)
    
    with open(out_path, "w") as f:
        for translation in translations:
            f.write(translation + "\n")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset', choices=['wmt21', 'wmt22'], required=True)
    parser.add_argument('--language-pair', choices=['de-en', 'en-de', 'en-ru', 'ru-en'], required=True)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument('--limit-segments', type=int, default=None,
                        help='Limit number of segments that are processed (used for testing)')
    args = parser.parse_args()
    
    out_path = main(args.testset, args.language_pair, args.beam_size, args.limit_segments)
    assert out_path.exists()
    print(f"Saved translations to {out_path}")
