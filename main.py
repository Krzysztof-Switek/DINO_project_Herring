"""Root-level entry point for OtolithDinoStandalone.

Usage:
    python main.py                   # train (domyślnie)
    python main.py --mode demo       # 1 epoka + pełny pipeline
    python main.py --mode report     # tylko raport HTML
"""
from src.entrypoint import main

if __name__ == "__main__":
    main()
