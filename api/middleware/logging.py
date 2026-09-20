"""
Request Logging Middleware
==========================
Logs incoming HTTP requests, response status codes, execution latency, and client IPs.
"""

import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response telemetry."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Log request start
        method = request.method
        url = request.url.path
        client_host = request.client.host if request.client else "unknown"

        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000
        status_code = response.status_code

        logger.info(
            f"[{request_id}] {method} {url} | Status: {status_code} | "
            f"Latency: {duration_ms:.1f}ms | Client: {client_host}"
        )
        response.headers["X-Request-ID"] = request_id
        return response
