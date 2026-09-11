import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel

PII_HELP_URL = "https://www.groww.in/help/privacy-data"
OPINION_URL = "https://www.groww.in/mutual-funds"
PERFORMANCE_URL = "https://www.groww.in/help/disclaimers"
OUT_OF_SCOPE_URL = "https://www.groww.in/mutual-funds"

REFUSAL_PII = (
    "I can't discuss personal or sensitive information. "
    "For your safety, please do not share PAN, Aadhaar, phone numbers, "
    "email addresses, or OTPs in this chat."
)
REFUSAL_OPINION = (
    "I only provide objective, fact-based information drawn from the fund "
    "documents in the corpus. I can't give personalised advice or "
    "recommendations about whether to buy, sell, or hold a fund."
)
REFUSAL_PERFORMANCE = (
    "I can share what the fund documents say about performance and returns, "
    "but past performance does not guarantee future results."
)
REFUSAL_OUT_OF_SCOPE = (
    "I can only answer questions about the HDFC mutual funds covered by the "
    "corpus. Your question falls outside that scope."
)

PAN_FULL_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
PAN_RE = re.compile(r"\b[A-Z]{4}[0-9]{2}\b")
AADHAAR_RE = re.compile(r"\b\d{12}\b")
AADHAAR_SPACED_RE = re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b")
PHONE_PLUS_RE = re.compile(r"\+\d{10,15}")
PHONE_RE = re.compile(r"\b(?:91[\s-]?)?[6-9]\d{9}\b")
PHONE_10_RE = re.compile(r"\b[6-9]\d{9}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
OTP_WORD_RE = re.compile(r"\b(otp|aadhaar|aadhar|pan number)\b", re.I)
OTP_CONTEXT_RE = re.compile(r"\b(enter|send|share|received|provide|is)\b", re.I)

OPINION_WORDS = [
    "should i buy",
    "should i invest",
    "is it a good",
    "recommend",
    "advice",
    "advise",
    "buy or sell",
    "hold or sell",
    "better to buy",
    "worth buying",
    "good time to",
]

OUT_OF_SCOPE_WORDS = [
    "etf",
    "crypto",
    "bitcoin",
    "fixed deposit",
    "cryptocurrency",
    "stock market",
    "share market",
    "sbi fund",
    "icici fund",
    "fd return",
]

OPINION_RE = re.compile(
    r"\b(should i buy|should i invest|would you recommend|do you recommend|"
    r"advise|advice|buy or sell|hold or sell|better to buy|worth buying|"
    r"good time to|is it a good)\b",
    re.I,
)

PERFORMANCE_WORDS = [
    "return",
    "returns",
    "performance",
    "xirr",
    "annualis",
    "annualiz",
    "outperform",
    "yield",
    "gave last year",
    "earn",
]
PERFORMANCE_RE = re.compile(
    r"\b(return|returns|performance|xirr|annualis|annualiz|outperform|yield)\b",
    re.I,
)


def last_updated_from(url: Optional[str] = None) -> str:
    if url:
        return f"\n\nLast updated from sources: {url}"
    return "\n\nLast updated from sources."


class GuardVerdict(str, Enum):
    PII = "pii"
    OPINION = "opinion"
    PERFORMANCE = "performance"
    OUT_OF_SCOPE = "out_of_scope"
    FACTUAL = "factual"


class GuardRefusal(BaseModel):
    verdict: GuardVerdict
    message: str
    url: Optional[str] = None


class QueryGuard:
    def check(self, question: str) -> Optional[GuardRefusal]:
        q = question.strip()
        if not q:
            return None
        if self._has_pii(q):
            return GuardRefusal(
                verdict=GuardVerdict.PII,
                url=PII_HELP_URL,
                message=REFUSAL_PII + last_updated_from(PII_HELP_URL),
            )
        if OPINION_RE.search(q):
            return GuardRefusal(
                verdict=GuardVerdict.OPINION,
                url=OPINION_URL,
                message=REFUSAL_OPINION + last_updated_from(OPINION_URL),
            )
        if self._is_out_of_scope(q):
            return GuardRefusal(
                verdict=GuardVerdict.OUT_OF_SCOPE,
                url=OUT_OF_SCOPE_URL,
                message=REFUSAL_OUT_OF_SCOPE + last_updated_from(OUT_OF_SCOPE_URL),
            )
        if self._is_performance(q):
            return GuardRefusal(
                verdict=GuardVerdict.PERFORMANCE,
                url=PERFORMANCE_URL,
                message=REFUSAL_PERFORMANCE + last_updated_from(PERFORMANCE_URL),
            )
        return None

    def _has_pii(self, question: str) -> bool:
        if PAN_FULL_RE.search(question) or AADHAAR_RE.search(question):
            return True
        if AADHAAR_SPACED_RE.search(question):
            return True
        if PHONE_PLUS_RE.search(question) or PHONE_10_RE.search(question):
            return True
        if EMAIL_RE.search(question):
            return True
        if OTP_WORD_RE.search(question) and OTP_CONTEXT_RE.search(question):
            return True
        return False

    def _is_performance(self, question: str) -> bool:
        lowered = question.lower()
        if PERFORMANCE_RE.search(lowered):
            return True
        return any(word in lowered for word in PERFORMANCE_WORDS)

    def _is_out_of_scope(self, question: str) -> bool:
        lowered = question.lower()
        if any(word in lowered for word in OUT_OF_SCOPE_WORDS):
            return True
        if re.search(r"\b(another fund|different fund|other than these funds)\b", lowered):
            return True
        return False
