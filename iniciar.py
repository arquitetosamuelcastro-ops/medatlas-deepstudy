#!/usr/bin/env python3
"""
MedAtlas DeepStudy — ponto de entrada
Uso:
  python start.py
  PORT=8742 ANTHROPIC_API_KEY=sk-ant-... python start.py
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8742))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=False)
