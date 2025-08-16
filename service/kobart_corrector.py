"""
KoBART EC 모델 래퍼
- 토크나이저/모델 로드
- "<SEG>" 같은 경계 토큰을 (필요 시) 추가
- 배치 인퍼런스를 위한 correct_batch 함수 제공
"""
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from typing import List
from config import KOBART_EC_PATH, SEG_TOKEN

class KoBARTCorrector:
    def __init__(self, model_name_or_path: str = KOBART_EC_PATH, device=None):
        # CUDA 가능하면 GPU 사용
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 토크나이저 로드
        self.tok = PreTrainedTokenizerFast.from_pretrained(model_name_or_path)

        # SEG 토큰이 vocab에 없다면 추가(파인튜닝 시에도 동일 설정 권장)
        added = 0
        if SEG_TOKEN not in self.tok.get_vocab():
            added = self.tok.add_tokens([SEG_TOKEN])

        # KoBART 모델 로드
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        if added > 0:
            self.model.resize_token_embeddings(len(self.tok))
        self.model.to(self.device).eval()

    @torch.inference_mode()  # 그래디언트 비활성화: 인퍼런스 최적화
    def correct_batch(self, inputs: List[str], max_new_tokens: int = 128) -> List[str]:
        """
        inputs: ["문장1 <SEG> 문장2 <SEG> 문장3", "다른배치 ..."]
        return: ["교정1 <SEG> 교정2 <SEG> 교정3", ...]
        """
        enc = self.tok(
            inputs,
            padding=True,        # 서로 다른 길이 배치 패딩
            truncation=True,     # 길이 과도한 입력은 안전하게 자름
            return_tensors="pt"
        ).to(self.device)

        # 요약형 모델의 과도한 재서술 방지를 위해 빔서치 파라미터 보수적 사용
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,  # 반복 억제
            num_beams=4              # 과도 탐색 방지(필요 시 1~4 사이 조정)
        )
        return self.tok.batch_decode(out, skip_special_tokens=True)