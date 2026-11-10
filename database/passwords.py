"""Password Hashing: turns login passwords into salted hashes and checks them.

Module      : passwords.py   (Design: Password Hashing)
Date        : Created 2026-10-19
Programmer  : Sebastian

DESCRIPTION
    Passwords are never stored. Registration stores a salted hash instead,
    and login checks a typed password against it. Uses only Python's
    standard library, so there is nothing extra to install.

        stored = hash_password("hunter22")
        verify_password("hunter22", stored)    # True
        verify_password("wrong", stored)       # False

IMPORTANT FUNCTIONS
    hash_password(password) -> str
        In : the plain password.
        Out: "pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>", safe to store.
    verify_password(password, stored) -> bool
        In : the typed password and the stored hash string.
        Out: True if they match. Never raises: a malformed or placeholder
             hash simply never matches.

IMPORTANT DATA STRUCTURES
    The stored hash string: algorithm, iteration count, salt and hash,
    separated by "$". It records how it was made, so the iteration count can
    be raised later without breaking existing accounts.

ALGORITHMS
    PBKDF2-HMAC-SHA256, 200,000 iterations, random 16-byte salt per password.
    The salt makes equal passwords hash differently and defeats precomputed
    tables; the iterations make each guess slow for an attacker.
    Chosen over bcrypt or Argon2 (stronger, memory-hard) because PBKDF2 is in
    the standard library and NIST-recommended (SP 800-132), so no extra
    package is needed; and over a plain SHA-256 hash, which is fast enough
    to brute-force.

    Constant-time comparison (hmac.compare_digest) takes the same time
    whichever byte differs, so response timing does not reveal how close a
    guess was.
"""

import hashlib
import hmac
import secrets

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 200_000


def hash_password(password):
    """Turns a plain password into a salted hash that is safe to store.

    Args:
        password (str): the plain password.

    Returns:
        str: "pbkdf2_sha256$200000$<salt hex>$<hash hex>".
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"{ALGORITHM}${ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    """Checks a password against a stored hash.

    Args:
        password (str): the password that was typed.
        stored (str): the hash from the database, made by hash_password.

    Returns:
        bool: True if they match; False for a wrong password or a malformed
        hash. Never raises.
    """
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != ALGORITHM:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
    except (ValueError, AttributeError):
        # Placeholder or malformed hash: it can never match.
        return False

    # compare_digest takes the same time whether the first byte or the last
    # one is wrong, so timing does not leak how close a guess was.
    return hmac.compare_digest(digest.hex(), digest_hex)
