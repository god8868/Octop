"""Recovery codes: generate, argon2id hash, atomic single-spend."""

from __future__ import annotations

import secrets
import string

from octop.infra.db.repos.user_recovery_codes import UserRecoveryCodeRepo
from octop.infra.users.password import hash_password, verify_password
from octop.infra.utils.offload import run_blocking

RECOVERY_CODE_COUNT = 10
_ALPHABET = string.ascii_uppercase + string.digits


def _one_code() -> str:
    # Dash-grouped: XXXX-XXXX
    raw = "".join(secrets.choice(_ALPHABET) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"


def generate_codes(count: int = RECOVERY_CODE_COUNT) -> list[str]:
    return [_one_code() for _ in range(count)]


def normalize_code(code: str) -> str:
    return (code or "").strip().upper().replace(" ", "").replace("-", "")


def codes_match(submitted: str, stored: str) -> bool:
    return normalize_code(submitted) == normalize_code(stored)


def _hash_codes(plain: list[str]) -> list[str]:
    """Hash a whole batch in one worker.

    Deliberately sequential. Ten parallel argon2 calls would overlap ten 64MiB
    working sets to shave ~500ms off a once-per-enrolment action; one hash at a
    time keeps the peak at 64MiB and still leaves the event loop free.
    """
    return [hash_password(normalize_code(c)) for c in plain]


def _find_match(needle: str, candidates: list[tuple[int, str]]) -> int | None:
    """Return the id of the row whose hash matches, or None.

    Takes plain tuples rather than ORM rows so nothing but the CPU work crosses
    the thread boundary.
    """
    for row_id, code_hash in candidates:
        if verify_password(needle, code_hash):
            return row_id
    return None


async def issue_codes(repo: UserRecoveryCodeRepo, user_id: int) -> list[str]:
    """Replace unused codes and return the plaintext set once.

    The ten hashes are ~660ms of CPU. Inline that was the longest single stall
    in the codebase: a person enrolling MFA froze every other request for the
    duration, and the response itself still took the same 660ms afterwards. The
    only thing that changes here is who waits.
    """
    plain = generate_codes()
    hashes = await run_blocking(_hash_codes, plain, label="issue_recovery_codes")
    repo.replace_all(user_id, hashes)
    return plain


async def try_spend_code(repo: UserRecoveryCodeRepo, user_id: int, code: str) -> bool:
    """Verify against unused hashes and atomically spend the matching row."""
    needle = normalize_code(code)
    if not needle:
        return False
    # A wrong code walks every unused hash — ten verifications, ~660ms. That is
    # the failure path, not an exotic one, so it needs the same treatment as the
    # success path rather than being written off as unlikely.
    candidates = [(row.id, row.code_hash) for row in repo.list_unused(user_id)]
    matched = await run_blocking(_find_match, needle, candidates, label="spend_recovery_code")
    if matched is None:
        return False
    return repo.try_spend(matched, user_id)
