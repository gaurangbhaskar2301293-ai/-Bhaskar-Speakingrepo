"""
Phoneme scoring utilities.

- Uses g2p_en to convert reference text -> ARPAbet-like phoneme sequence.
- Compares recognized phoneme sequence (from whisperx.get_phonemes) with reference phonemes
  using edit distance to compute phoneme_error_rate (PER).

Requirements:
  pip install g2p_en
"""

from typing import List, Tuple
import re

def _levenshtein(a: List[str], b: List[str]) -> int:
    # classic dynamic programming
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m]

def reference_to_phonemes(text: str) -> List[str]:
    """
    Convert text -> phonemes using g2p_en.
    Returns list of phoneme tokens (uppercase ARPAbet-like).
    """
    try:
        from g2p_en import G2p
    except Exception as e:
        raise RuntimeError("g2p_en not installed. pip install g2p_en") from e
    g2p = G2p()
    toks = g2p(text)
    # g2p returns mixture of phonemes and punctuation/words; filter
    phonemes = [t for t in toks if re.match(r"^[A-Za-z0-9]+$", t)]
    # Normalize (uppercase)
    phonemes = [p.upper() for p in phonemes]
    return phonemes

def compute_phoneme_error_rate(reference_text: str, recognized_phonemes: List[str]) -> float:
    """
    reference_text: the expected transcript text
    recognized_phonemes: list of phoneme strings from ASR alignment
    Returns phoneme_error_rate = edits / len(reference_phonemes)
    """
    ref_ph = reference_to_phonemes(reference_text)
    if len(ref_ph) == 0:
        return 1.0
    edits = _levenshtein(ref_ph, [p.upper() for p in recognized_phonemes])
    per = edits / len(ref_ph)
    return per
