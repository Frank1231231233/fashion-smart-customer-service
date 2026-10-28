"""导入示例商品:单入口同时写 SQLite 与 Chroma 向量库。

MD5 内容去重,可重复执行:内容未变的商品会被跳过。
运行: uv run python scripts/import_products.py
"""

from app.config import get_settings
from app.nodes.retrieval import get_product_vector_store
from app.retrieval.store import ProductStore

SAMPLES: list[dict[str, object]] = [
    {
        "product_id": "SKU-001",
        "title": "黑色修身连衣裙",
        "category": "连衣裙",
        "color": "黑色",
        "sizes": ["S", "M", "L"],
        "price": "299.00",
        "style": "通勤",
        "description": "聚酯纤维混纺面料,微弹修身剪裁,适合面试与日常通勤。",
    },
    {
        "product_id": "SKU-002",
        "title": "黑色真丝晚礼连衣裙",
        "category": "连衣裙",
        "color": "黑色",
        "sizes": ["S", "M"],
        "price": "829.00",
        "style": "宴会",
        "description": "桑蚕丝面料,垂坠感强,露背设计,适合晚宴与正式场合。",
    },
    {
        "product_id": "SKU-003",
        "title": "红色法式复古连衣裙",
        "category": "连衣裙",
        "color": "红色",
        "sizes": ["XS", "S", "M", "L"],
        "price": "459.00",
        "style": "复古",
        "description": "雪纺面料,方领泡泡袖,A字裙摆,适合约会与出游。",
    },
    {
        "product_id": "SKU-004",
        "title": "白色免烫衬衫",
        "category": "衬衫",
        "color": "白色",
        "sizes": ["S", "M", "L", "XL"],
        "price": "129.00",
        "style": "通勤",
        "description": "纯棉免烫工艺,版型挺括,机洗不易皱,职场基础款。",
    },
    {
        "product_id": "SKU-005",
        "title": "深蓝直筒牛仔裤",
        "category": "牛仔裤",
        "color": "深蓝色",
        "sizes": ["26", "27", "28", "29", "30"],
        "price": "199.00",
        "style": "休闲",
        "description": "棉弹牛仔布,中腰直筒版型,水洗工艺,百搭日常款。",
    },
    {
        "product_id": "SKU-006",
        "title": "灰色羊毛西装套装",
        "category": "西装",
        "color": "灰色",
        "sizes": ["S", "M", "L"],
        "price": "899.00",
        "style": "商务",
        "description": "羊毛混纺面料,含西装外套与西裤,修身商务剪裁。",
    },
    {
        "product_id": "SKU-007",
        "title": "米色A字半身裙",
        "category": "半身裙",
        "color": "米色",
        "sizes": ["S", "M", "L"],
        "price": "259.00",
        "style": "通勤",
        "description": "斜纹棉面料,高腰A字版型,及膝长度,温柔百搭。",
    },
    {
        "product_id": "SKU-008",
        "title": "卡其色中长款风衣",
        "category": "风衣",
        "color": "卡其色",
        "sizes": ["M", "L", "XL"],
        "price": "699.00",
        "style": "英伦",
        "description": "棉质华达呢,防风防泼水,经典双排扣束腰设计。",
    },
    {
        "product_id": "SKU-009",
        "title": "白色纯棉短袖T恤",
        "category": "T恤",
        "color": "白色",
        "sizes": ["S", "M", "L", "XL", "XXL"],
        "price": "79.00",
        "style": "基础",
        "description": "100%精梳棉,圆领基础款,亲肤透气,多色可选。",
    },
    {
        "product_id": "SKU-010",
        "title": "酒红色改良旗袍",
        "category": "旗袍",
        "color": "酒红色",
        "sizes": ["S", "M", "L"],
        "price": "569.00",
        "style": "中式",
        "description": "丝绒面料,改良短款剪裁,盘扣立领,日常可穿的新中式。",
    },
]


def main() -> None:
    settings = get_settings()
    store = ProductStore(
        settings.product_db_path,
        document_index=get_product_vector_store(),
    )
    report = store.import_products(SAMPLES)
    print(f"导入完成: {report}")


if __name__ == "__main__":
    main()
