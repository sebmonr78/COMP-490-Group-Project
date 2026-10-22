"""Password hashing for DELTA logins.

Uses PBKDF2 from Python's standard library, so there is nothing extra to
install. Stored hashes look like:

    pbkdf2_sha256$200000$<salt hex>$<hash hex>

    import passwords

    stored = passwords.hash_password("hunter22")
    passwords.verify_password("hunter22", stored)   # True
"""

import hashlib
import hmac
import secrets

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 200_000


def hash_password(password):
    """Turn a plain password into a salted hash that is safe to store."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"{ALGORITHM}${ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    """True if the password matches the stored hash. Never raises on bad input."""
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
