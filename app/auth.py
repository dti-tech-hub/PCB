"""Authentication: bcrypt passwords + signed session cookie (itsdangerous).

Pages other than /login and /register require a valid session.
"""

from typing import Optional

import bcrypt
from fastapi import Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app import database as db
from app.config import SECRET_KEY, SESSION_COOKIE, SESSION_MAX_AGE


def _serializer() -> URLSafeTimedSerializer:
    """Serializer for signing session cookies."""
    return URLSafeTimedSerializer(SECRET_KEY, salt="pcb-session")


# --- Passwords ---

def hash_password(password: str) -> str:
    """Hash a plain password with bcrypt. Returns utf-8 string."""
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Check a plain password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"),
                              password_hash.encode("utf-8"))
    except Exception:
        return False


# --- Sessions ---

def create_session_token(user_id: int) -> str:
    """Create a signed token carrying the user id."""
    return _serializer().dumps({"uid": int(user_id)})


def decode_session_token(token: str) -> Optional[int]:
    """Return user id from token, or None if invalid/expired."""
    try:
        data = _serializer().loads(token, max_age=SESSION_MAX_AGE)
        return int(data.get("uid"))
    except (BadSignature, SignatureExpired, Exception):
        return None


def get_current_user(request: Request) -> Optional[dict]:
    """Return logged-in user dict, or None."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    user_id = decode_session_token(token)
    if user_id is None:
        return None
    return db.get_user_by_id(user_id)


def login_response_redirect(username: str, password: str):
    """Validate credentials. Returns (ok, user_or_error)."""
    if not username or not password:
        return False, "Username and password are required."
    user = db.get_user_by_username(username.strip())
    if user is None:
        return False, "Invalid username or password."
    if not verify_password(password, user["password_hash"]):
        return False, "Invalid username or password."
    return True, user


def validate_registration(username: str, password: str,
                          confirm: str) -> Optional[str]:
    """Return error string, or None if registration input is valid."""
    u = (username or "").strip()
    if len(u) < 3:
        return "Username must be at least 3 characters."
    if len(u) > 32:
        return "Username must be at most 32 characters."
    if not u.replace("_", "").replace("-", "").isalnum():
        return "Username may contain only letters, numbers, _ and -."
    if len(password or "") < 4:
        return "Password must be at least 4 characters."
    if password != confirm:
        return "Passwords do not match."
    if db.get_user_by_username(u) is not None:
        return "Username already taken. Please choose another."
    return None
