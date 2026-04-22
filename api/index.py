"""
Vercel serverless entry point — 將 FastAPI app 暴露給 Vercel Python runtime。
Vercel 會把所有 /api/* 的請求路由到這裡，再由 FastAPI 處理。
"""
import sys
import os

# 確保根目錄在 Python path 中，讓 backend.main 可以被 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app  # noqa: F401 — Vercel ASGI handler 需要這個 app 物件
