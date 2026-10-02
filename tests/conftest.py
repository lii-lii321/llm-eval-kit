"""共享夹具：小型玩具语料（3 段中文短文）。"""

import pytest

from llm_eval_kit.corpus import Doc

TOY_TEXTS = {
    "d01": "BM25 是经典的稀疏检索算法，基于词频与逆文档频率对文档打分，不需要训练模型。",
    "d02": "向量数据库存储文本向量，支持近似最近邻检索，常见引擎包括 Faiss 与 Milvus。",
    "d03": "微调把领域数据继续训练进模型参数，LoRA 等参数高效微调适合固定风格与格式任务。",
}


@pytest.fixture
def toy_docs() -> list[Doc]:
    return [Doc(doc_id=doc_id, text=text, source="toy") for doc_id, text in TOY_TEXTS.items()]
