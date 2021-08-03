# -*- coding: utf-8 -*-

import os
import unicodedata

from nltk.tokenize.api import TokenizerI
from nltk.tokenize.texttiling import TextTilingTokenizer
import nltk

from hanziconv.hanziconv import HanziConv
import jieba

def is_punctuation_mark(ch: str) -> bool:
    if unicodedata.category(ch)[0] == 'P':
        return True
    return False


class ChineseSentenceSegmenter(TokenizerI):
    def tokenize(self, text):
        sent = ""
        delimiters = {'。', '！', '？'}
        for ch in list(text):
            sent += ch
            if ch in delimiters and sent:
                yield sent
                sent = ""
        if sent:
            yield sent


class ChineseWordSegmenter(TokenizerI):
    def tokenize(self, text):
        #   better Chinese word segmentation can be used.
        for tok in jieba.cut(text):
            yield tok


class ChineseTextTilingTokenizer(TextTilingTokenizer):
    def __init__(self):
        super().__init__(stopwords=set(), w=40, k=20)
        self.cws_model = ChineseWordSegmenter()
        self.css_model = ChineseSentenceSegmenter()

    def tokenize(self, text):
        text = HanziConv.toSimplified(text)
        sents = []
        for raw_sent in self.css_model.tokenize(text):
            sent = " ".join(list(self.cws_model.tokenize(raw_sent))).strip()
            sents.append(sent)
        text = "\n\n".join(sents)

        lowercase_text = text.lower()
        paragraph_breaks = self._mark_paragraph_breaks(text)
        text_length = len(lowercase_text)

        # Tokenization step starts here

        # Remove punctuation
        nopunct_text = ''.join(
            c for c in lowercase_text if not is_punctuation_mark(c)
        )
        nopunct_par_breaks = self._mark_paragraph_breaks(nopunct_text)

        tokseqs = self._divide_to_tokensequences(nopunct_text)

        # The morphological stemming step mentioned in the TextTile
        # paper is not implemented.  A comment in the original C
        # implementation states that it offers no benefit to the
        # process. It might be interesting to test the existing
        # stemmers though.
        # words = _stem_words(words)

        # Filter stopwords
        for ts in tokseqs:
            ts.wrdindex_list = [
                wi for wi in ts.wrdindex_list if len(wi[0]) > 1 and wi[0] not in self.stopwords
            ]

        token_table = self._create_token_table(tokseqs, nopunct_par_breaks)
        # End of the Tokenization step

        gap_scores = self._block_comparison(tokseqs, token_table)
        smooth_scores = self._smooth_scores(gap_scores)
        depth_scores = self._depth_scores(smooth_scores)
        segment_boundaries = self._identify_boundaries(depth_scores)
        normalized_boundaries = self._normalize_boundaries(
            text, segment_boundaries, paragraph_breaks
        )
        segmented_text = []
        prevb = 0
        for b in normalized_boundaries:
            if b == 0:
                continue
            segmented_text.append(text[prevb:b])
            prevb = b
        if prevb < text_length:  # append any text that may be remaining
            segmented_text.append(text[prevb:])
        if not segmented_text:
            segmented_text = [text]
        if self.demo_mode:
            return gap_scores, smooth_scores, depth_scores, segment_boundaries
        return segmented_text


if __name__ == "__main__":
    tokenizer = ChineseTextTilingTokenizer()
    with open(os.path.join("data", "sample.txt")) as fin:
        text = fin.read()
    segs = list(tokenizer.tokenize(text))
    print(len(segs))
    for seg in segs:
        print(seg)
        print("\n")


