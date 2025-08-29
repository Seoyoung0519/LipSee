# service/rerank.py
import torch

@torch.inference_mode()
def sent_nll(tok, model, text: str, device=None, max_length: int = 256) -> float:
    """
    단일 문장에 대한 NLL(loss) 계산. BART계열은 token_type_ids 미지원 → 제거 필수.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch = tok([text], return_tensors="pt", padding=True, truncation=True,
                max_length=max_length, return_token_type_ids=False)
    # 혹시라도 들어오면 버림
    batch = {k: v.to(device) for k, v in batch.items() if k != "token_type_ids"}

    out = model(**batch, labels=batch["input_ids"])
    loss = out.loss
    return float(loss.detach().cpu().item())


@torch.inference_mode()
def choose_with_gain(
    base_text: str,
    candidates: list,
    tok,
    model,
    device=None,
    *,
    tau_gain: float = 0.05,   # 개선도 임계치(= aggressive_threshold)
    tau_len: int = 3,         # (호환용, 여기선 미사용)
    length_penalty: float = 0.0,
    max_length: int = 256,
):
    """
    base_text 대비 각 candidate의 NLL 개선도(gain)를 계산해서 최선 후보를 선택.
    gain = base_loss - cand_loss - length_penalty * max(0, len(cand) - len(base))
    반환: (chosen_text, picked_candidate, gain)
    - gain < tau_gain 이면 원문 유지
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # base 문장 점수
    base_loss = sent_nll(tok, model, base_text, device=device, max_length=max_length)

    best_text = base_text
    best_cand = None
    best_gain = 0.0

    if candidates is None:
        candidates = []

    for cand in candidates:
        if not cand or cand == base_text:
            continue
        cand_loss = sent_nll(tok, model, cand, device=device, max_length=max_length)
        len_pen = length_penalty * max(0, len(cand) - len(base_text))
        gain = base_loss - cand_loss - len_pen
        if gain > best_gain:
            best_gain = float(gain)
            best_text = cand
            best_cand = cand

    if best_gain < float(tau_gain):
        return base_text, None, 0.0

    return best_text, best_cand, best_gain
