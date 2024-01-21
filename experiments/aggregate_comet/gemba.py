import json
import os
from pathlib import Path

import lmql
from sqlitedict import SqliteDict


class GembaMetric:

    def __init__(self, model: str = "openai/babbage-002"):
        self.model = model

    def score(self, src: str, ref: str, hyp: str, src_lang: str, tgt_lang: str) -> int:
        """
        :param src: Source sentence
        :param ref: Human reference translation
        :param hyp: Machine translation
        :param src_lang: Source language
        :param tgt_lang: Target language
        :return: Score between 0 and 100, where 0 means "no meaning preserved" and 100 means "perfect meaning and grammar"
        """
        with self.load_cache() as cache:
            score = cache.get(f"{src}_{ref}_{hyp}_{src_lang}_{tgt_lang}", None)
            if score is None:
                score = gemba(src, ref, hyp, src_lang, tgt_lang, model=lmql.model(self.model))
                cache[f"{src}_{ref}_{hyp}_{src_lang}_{tgt_lang}"] = score
                cache.commit()
        return score

    @property
    def cache_path(self) -> Path:
        """
        :return: Path of the SQLite database where the translations and scores are cached
        """
        cache_dir = Path(os.getenv("GEMBA_CACHE", Path.home() / ".cache" / "gemba" / self.model.replace("/", "_")))
        if not cache_dir.exists():
            os.mkdir(cache_dir)
        return cache_dir / (str(self).replace("/", "_") + ".sqlite")

    def load_cache(self) -> SqliteDict:
        """
        :return: A connection to the SQLite database where the translations and scores are cached
        """
        return SqliteDict(self.cache_path, timeout=15, encode=json.dumps, decode=json.loads)


@lmql.query(model="openai/babbage-002", decoder="argmax", max_len=500)
def gemba(src: str, ref: str, hyp: str, src_lang: str, tgt_lang: str):
    '''lmql
    """Score the following translation from {src_lang} to {tgt_lang} with respect to the human reference on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" and a score of one hundred means "perfect meaning and grammar".

    {src_lang} source: "{src}"
    {tgt_lang} human reference: "{ref}"
    {tgt_lang} machine translation: "{hyp}"
    Score: [SCORE]
    """ where SCORE in set(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100"])
    return SCORE
    '''


if __name__ == '__main__':
    score = gemba(
        src="Dies ist die perfekte Übersetzung.",
        ref="This is the perfect translation.",
        hyp="This is a near-perfect translation.",
        src_lang="German",
        tgt_lang="English",
    )
    print(score)
    score = gemba(
        src="Dies ist die perfekte Übersetzung.",
        ref="This is the perfect translation.",
        hyp="I've never seen a worse translation.",
        src_lang="German",
        tgt_lang="English",
    )
    print(score)

    from lmql.runtime.bopenai import get_stats
    print(get_stats())