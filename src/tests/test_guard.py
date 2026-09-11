import numpy as np

from src.retrieve.guard import GuardRefusal, GuardVerdict, QueryGuard

guard = QueryGuard()


def _assert_refusal(question: str, verdict: GuardVerdict) -> GuardRefusal:
    rej = guard.check(question)
    assert rej is not None, f"expected {verdict.name} refusal for {question!r}"
    assert rej.verdict is verdict, f"expected {verdict.name}, got {rej.verdict}"
    assert rej.message, "refusal needs a message"
    assert rej.url, "refusal needs a help url"
    return rej


def test_guard_allows_factual_question() -> None:
    assert guard.check("What is the expense ratio of the HDFC Flexi Cap fund?") is None


def test_guard_allows_plan_type() -> None:
    assert guard.check("Is the HDFC Flexi Cap fund a direct growth plan?") is None


def test_guard_allows_fund_comparison() -> None:
    q = "Compare the HDFC Small Cap and HDFC Large Cap fund categories."
    assert guard.check(q) is None


def test_guard_refuses_pan() -> None:
    _assert_refusal("My PAN is ABCDE1234F, is it valid?", GuardVerdict.PII)


def test_guard_refuses_aadhaar() -> None:
    _assert_refusal("Here is my aadhaar: 1234 5678 9012.", GuardVerdict.PII)


def test_guard_refuses_phone() -> None:
    _assert_refusal("Call me on +91 9876543210 about HDFC funds.", GuardVerdict.PII)


def test_guard_refuses_email() -> None:
    _assert_refusal("Email the answer to me@example.com.", GuardVerdict.PII)


def test_guard_refuses_otp() -> None:
    _assert_refusal("I got an OTP, where do I enter it?", GuardVerdict.PII)


def test_guard_refuses_opinion() -> None:
    rej = _assert_refusal(
        "Should I buy the HDFC Flexi Cap fund this year?", GuardVerdict.OPINION
    )
    assert "Last updated from sources" in rej.message


def test_guard_refuses_performance() -> None:
    _assert_refusal(
        "How much did the HDFC Small Cap return last year?", GuardVerdict.PERFORMANCE
    )


def test_guard_refuses_out_of_scope() -> None:
    _assert_refusal("Does the HDFC fund outperform ETFs?", GuardVerdict.OUT_OF_SCOPE)


def test_guard_empty_question_returns_none() -> None:
    assert guard.check("   ") is None
