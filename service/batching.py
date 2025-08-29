from typing import List
from config import CHUNK_CHAR_LIMIT, SEG_TOKEN

def make_batches(texts: List[str]) -> List[List[str]]:
    batches=[]
    cur=[]
    cur_len=0
    for t in texts:
        l=len(t)
        if cur and cur_len + l > CHUNK_CHAR_LIMIT:
            batches.append(cur); cur=[]; cur_len=0
        cur.append(t); cur_len += l
    if cur: batches.append(cur)
    return batches