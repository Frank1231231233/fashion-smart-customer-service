"""模拟订单服务:阶段 6 真实调用验收用。

启动:uvicorn scripts.mock_order_service:app --port 9000
"""

from fastapi import FastAPI, Header, HTTPException

app = FastAPI(title="Mock Order Service", version="1.0.0")

_ORDERS: dict[str, list[dict[str, str]]] = {
    "user-001": [
        {
            "order_id": "ORD-001",
            "status": "shipped",
            "amount": "299.00",
            "phone": "13800000000",
            "address": "上海市测试路 1 号",
        },
        {
            "order_id": "ORD-002",
            "status": "paid",
            "amount": "129.50",
            "phone": "13800000000",
            "address": "上海市测试路 1 号",
        },
    ],
    "user-002": [
        {
            "order_id": "ORD-101",
            "status": "completed",
            "amount": "89.00",
            "phone": "13900000000",
            "address": "北京市示例街 2 号",
        },
    ],
}


@app.get("/internal/orders")
def list_orders(x_user_id: str = Header()):
    return _ORDERS.get(x_user_id, [])


@app.get("/internal/orders/{order_id}")
def get_order_detail(order_id: str, x_user_id: str = Header()):
    for order in _ORDERS.get(x_user_id, []):
        if order["order_id"] == order_id:
            return order
    raise HTTPException(status_code=404, detail="order not found")
