from kiwipiepy import Kiwi

_kiwi = Kiwi()

# NNG/NNP 범위의 명사 span(문자단위 start,end) 뽑기
def noun_spans(text: str):
    spans = []
    off = 0
    for token in _kiwi.tokenize(text, normalize_coda=False):
        form = token.form
        start = token.start
        end = start + len(form)
        tag = token.tag
        if tag in ("NNG", "NNP"):  # 보통명사/고유명사
            spans.append((start, end))
    return spans

