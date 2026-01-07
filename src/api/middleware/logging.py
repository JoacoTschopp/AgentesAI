"""
Logging Middleware - Request/Response logging for FastAPI.

This module provides structured logging for all API requests,
including timing, status codes, and request metadata.
"""

import time
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.
    
    Logs request details, response status, and processing time
    for all API endpoints.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.
        
        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in chain.
            
        Returns:
            HTTP response.
        """
        start_time = time.time()
        
        request_id = request.headers.get("X-Request-ID", "")
        
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            request_id=request_id,
        )

        try:
            response = await call_next(request)
            
            process_time = time.time() - start_time
            
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                process_time_ms=round(process_time * 1000, 2),
                request_id=request_id,
            )
            
            response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
            
            return response

        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                process_time_ms=round(process_time * 1000, 2),
                request_id=request_id,
            )
            raise
