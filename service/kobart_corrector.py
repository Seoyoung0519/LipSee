import os, json, copy, torch, re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, PreTrainedTokenizerFast
from safetensors.torch import load_file
from transformers import BartConfig, BartForConditionalGeneration

from .postprocess import postprocess_text, sent_split, light_normalize
from config import (
    KOBART_EC_PATH, NUM_BEAMS, NO_REPEAT_NGRAM, REPETITION_PENALTY, LENGTH_PENALTY,
    GUARD_CER, GUARD_MAX_SENT
)
import jiwer

class GuardRails:
    def __init__(self, cer_gate: float = GUARD_CER, max_sent_cap: int = GUARD_MAX_SENT):
        self.cer_gate = cer_gate
        self.max_sent_cap = max_sent_cap
        self.last_sent = None

    def apply(self, src: str, pred_pp: str) -> str:
        # 1) 수정폭/문장수 게이트
        cer = jiwer.cer([src], [pred_pp])
        if cer > self.cer_gate or len(sent_split(pred_pp)) > self.max_sent_cap:
            return light_normalize(src)

        # 2) 직전 마지막 문장과 중복 제거
        parts = sent_split(pred_pp)
        if parts and self.last_sent and parts[-1] == self.last_sent:
            parts = parts[:-1]
        out = " ".join(parts) if parts else pred_pp

        # 3) 짧은 '고아 꼬리' 컷
        parts2 = sent_split(out)
        if len(parts2) >= 2:
            tail = parts2[-1].strip()
            # 원문 첫 토큰(띄어쓰기 기준)
            first_tok = re.split(r'[\s,.;!?…]+', src.strip())[0] if src.strip() else ""
            # 길이가 매우 짧고, 문장부호로 끝나지 않으며,
            # (a) 원문에 거의 그대로 존재하거나 (b) 원문 첫 토큰의 접두면 → 꼬리로 판단
            if (
                len(tail) <= 3 and
                not re.search(r'[.!?…]$', tail) and
                (tail in src or (first_tok and first_tok.startswith(tail)))
            ):
                parts2 = parts2[:-1]
                out = " ".join(parts2).strip()

        # 4) last_sent 갱신
        parts3 = sent_split(out)
        self.last_sent = parts3[-1] if parts3 else self.last_sent
        return out.strip()

def _manual_load_tokenizer(model_dir):
    tj  = os.path.join(model_dir, "tokenizer.json")
    stm = os.path.join(model_dir, "special_tokens_map.json")
    tkc = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(tj):
        raise FileNotFoundError(f"tokenizer.json이 {model_dir}에 없습니다.")
    tok_kwargs = {}
    if os.path.exists(stm):
        tok_kwargs.update(json.load(open(stm, encoding="utf-8")))
    if os.path.exists(tkc):
        cfg = json.load(open(tkc, encoding="utf-8"))
        for k in ["bos_token","eos_token","pad_token","unk_token","sep_token","cls_token","mask_token","additional_special_tokens"]:
            if k in cfg: tok_kwargs[k] = cfg[k]
    return PreTrainedTokenizerFast(tokenizer_file=tj, **tok_kwargs)

def _manual_load_model(model_dir, device):
    cfg_dict = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    config = BartConfig(**cfg_dict)
    model  = BartForConditionalGeneration(config)
    state  = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()

class KoBARTCorrector:
    def __init__(self, model_dir=KOBART_EC_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 토크나이저
        try:
            self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        except Exception:
            self.tok = _manual_load_tokenizer(model_dir)
        # 모델
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=torch.float32
            )
        except Exception:
            self.model = _manual_load_model(model_dir, self.device)
        self.model.to(self.device).eval()

        # generation config
        try:
            self.base_gen = GenerationConfig.from_pretrained(model_dir, local_files_only=True)
        except Exception:
            self.base_gen = GenerationConfig(
                num_beams=NUM_BEAMS, num_beam_groups=1, num_return_sequences=1,
                no_repeat_ngram_size=NO_REPEAT_NGRAM, encoder_no_repeat_ngram_size=NO_REPEAT_NGRAM,
                repetition_penalty=REPETITION_PENALTY, length_penalty=LENGTH_PENALTY,
                early_stopping=True, renormalize_logits=True
            )
        self.base_gen.eos_token_id = self.tok.eos_token_id
        self.base_gen.pad_token_id = self.tok.pad_token_id or self.tok.eos_token_id
        self.bad_words_ids = self.tok(["!!!!","???","ㅋㅋㅋㅋ","ㅎㅎㅎㅎ"], add_special_tokens=False).input_ids
        self.rails = GuardRails()

    @torch.inference_mode()
    def correct_batch(self, texts):
        # 배치마다 last_sent 초기화(세션 간 중복 오판 방지)
        self.rails.last_sent = None

        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_token_type_ids=False
        )
        enc = {k: v.to(self.device) for k, v in enc.items() if k != "token_type_ids"}

        gen = copy.deepcopy(self.base_gen)
        gen.bad_words_ids = self.bad_words_ids

        # 입력 길이 기반 상한: ~1.2배, [32, 96] 클램프
        input_len = int(enc["input_ids"].shape[1])
        gen.max_new_tokens = min(96, max(32, int(input_len * 1.2)))

        out = self.model.generate(**enc, generation_config=gen)
        preds = self.tok.batch_decode(out, skip_special_tokens=True)

        finals = []
        for src, p in zip(texts, preds):
            n_sent = max(1, len(sent_split(src)))
            # ❗ n_src 인자명으로 수정
            pp = postprocess_text(p, n_src=n_sent, max_chars=160)
            finals.append(self.rails.apply(src, pp))
        return finals
