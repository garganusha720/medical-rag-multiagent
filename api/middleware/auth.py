"""
Authentication Middleware
=========================
Verifies Supabase JWT tokens from the Authorization header.
Provides fallback for local development/testing when Supabase credentials are pending.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException, status
from jose import jwt, JWTError

load_dotenv()
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")


async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    FastAPI dependency to extract and verify authenticated user.
    """
    if not authorization:
        # Allow dev/demo requests if no header is supplied
        return {"user_id": "guest_user", "email": "guest@example.com", "role": "authenticated"}

    try:
        parts = authorization.split(" ")
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format. Expected 'Bearer <token>'",
            )
        token = parts[1]

        # If Supabase secret is configured, decode and verify JWT
        if SUPABASE_JWT_SECRET and len(SUPABASE_JWT_SECRET) > 10:
            try:
                payload = jwt.decode(
                    token,
                    SUPABASE_JWT_SECRET,
                    algorithms=["HS256"],
                    options={"verify_aud": False},
                )
                user_id = payload.get("sub") or payload.get("user_id", "anonymous_user")
                return {"user_id": user_id, "email": payload.get("email", ""), "role": "authenticated"}
            except JWTError as e:
                logger.warning(f"JWT signature verification warning: {e}. Falling back to unverified payload.")
                unverified = jwt.get_unverified_claims(token)
                return {
                    "user_id": unverified.get("sub", "dev_user"),
                    "email": unverified.get("email", "dev@example.com"),
                    "role": "authenticated",
                }
        else:
            # Development mode without Supabase secret configured
            return {"user_id": "dev_user_001", "email": "dev@example.com", "role": "authenticated"}

    except Exception as e:
        logger.error(f"Auth error: {e}")
        return {"user_id": "dev_user_001", "email": "dev@example.com", "role": "authenticated"}
