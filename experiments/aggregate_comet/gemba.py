import asyncio
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
        :return: Score between 0 and 9, where 0 means "no meaning preserved" and 9 means "perfect meaning and grammar"
        """
        with self.load_cache() as cache:
            score = cache.get(f"{src}_{ref}_{hyp}_{src_lang}_{tgt_lang}", None)

            if score is None:

                def async_to_sync(awaitable):
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(awaitable)

                score = async_to_sync(gemba(src, ref, hyp, src_lang, tgt_lang, model=lmql.model(self.model)))

                cache[f"{src}_{ref}_{hyp}_{src_lang}_{tgt_lang}"] = score
                cache.commit()
        return score

    @property
    def cache_path(self) -> Path:
        """
        :return: Path of the SQLite database where the translations and scores are cached
        """
        cache_dir = Path(os.getenv("GEMBA_CACHE", Path.home() / ".cache" / "gemba"))
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        return cache_dir / (self.model.replace("/", "_") + ".sqlite")

    def load_cache(self) -> SqliteDict:
        """
        :return: A connection to the SQLite database where the translations and scores are cached
        """
        return SqliteDict(self.cache_path, timeout=15, encode=json.dumps, decode=json.loads)


@lmql.query(model="openai/babbage-002", decoder="argmax", max_len=500)
def gemba(src: str, ref: str, hyp: str, src_lang: str, tgt_lang: str):
    '''lmql
    """Score the following translation from {src_lang} to {tgt_lang} with respect to the human reference on a continuous scale from 0 to 9, where a score of zero means "no meaning preserved" and a score of nine means "perfect meaning and grammar".

    {src_lang} source: "{src}"
    {tgt_lang} human reference: "{ref}"
    {tgt_lang} machine translation: "{hyp}"
    Score: [SCORE]
    """ where INT(SCORE)
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
