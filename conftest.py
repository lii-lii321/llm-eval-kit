"""让 pytest 在包未安装的情况下也能从 src 目录导入 llm_eval_kit。"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
