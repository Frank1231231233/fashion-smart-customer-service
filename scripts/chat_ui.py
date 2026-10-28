"""多轮对话 UI(演示用)。

启动: uv run streamlit run scripts/chat_ui.py
依赖后端运行在 http://127.0.0.1:8000。
"""

import httpx
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
TIMEOUT = 90.0

st.set_page_config(page_title="服装智能客服")

if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "flash" not in st.session_state:
    st.session_state.flash = None


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def _logout() -> None:
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.thread_id = None
    st.session_state.messages = []


def _login(username: str, password: str) -> str | None:
    response = httpx.post(
        f"{API_BASE}/auth/login",
        json={"username": username, "password": password},
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        return None
    body = response.json()
    st.session_state.token = body["token"]
    st.session_state.user = body["user"]
    return body["user"]


def render_auth() -> None:
    st.title("服装智能客服")
    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        username = st.text_input("用户名", key="login_username")
        password = st.text_input("密码", type="password", key="login_password")
        if st.button("登录", use_container_width=True):
            try:
                if _login(username.strip(), password):
                    st.rerun()
                else:
                    st.error("用户名或密码错误")
            except httpx.HTTPError:
                st.error("后端服务不可用,请先启动 app.main")

    with register_tab:
        username = st.text_input("用户名(2-32 位)", key="reg_username")
        password = st.text_input("密码(至少 6 位)", type="password", key="reg_password")
        role = st.selectbox("角色", ["customer", "operator"], format_func=lambda r: "用户" if r == "customer" else "客服")
        if st.button("注册并登录", use_container_width=True):
            try:
                response = httpx.post(
                    f"{API_BASE}/auth/register",
                    json={
                        "username": username.strip(),
                        "password": password,
                        "role": role,
                    },
                    timeout=TIMEOUT,
                )
                if response.status_code == 409:
                    st.error("用户名已被占用")
                elif response.status_code != 201:
                    st.error(f"注册失败:{response.json().get('detail', response.text)}")
                elif _login(username.strip(), password):
                    st.rerun()
            except httpx.HTTPError:
                st.error("后端服务不可用,请先启动 app.main")


def _switch_chat(thread_id: str | None) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.messages = []
    if thread_id is not None:
        response = httpx.get(
            f"{API_BASE}/chats/{thread_id}/messages",
            headers=_headers(),
            timeout=TIMEOUT,
        )
        if response.status_code == 200:
            st.session_state.messages = response.json()["messages"]


def render_customer() -> None:
    st.title("服装智能客服")

    with st.sidebar:
        user = st.session_state.user
        st.header(f"你好,{user['username']}")
        if st.button("退出登录", use_container_width=True):
            _logout()
            st.rerun()

        if st.button("新建会话", use_container_width=True):
            _switch_chat(None)
            st.rerun()

        st.header("历史会话")
        try:
            chats = httpx.get(
                f"{API_BASE}/chats",
                headers=_headers(),
                timeout=TIMEOUT,
            ).json()["chats"]
        except httpx.HTTPError:
            st.error("后端服务不可用,请先启动 app.main")
            chats = []

        for chat in chats:
            is_current = chat["thread_id"] == st.session_state.thread_id
            if st.button(
                f"● {chat['title']}" if is_current else chat["title"],
                key=chat["thread_id"],
                use_container_width=True,
            ):
                _switch_chat(chat["thread_id"])
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("请输入你的问题")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        payload: dict[str, str] = {"message": prompt}
        if st.session_state.thread_id:
            payload["thread_id"] = st.session_state.thread_id

        with st.chat_message("assistant"):
            with st.spinner("客服正在输入..."):
                try:
                    response = httpx.post(
                        f"{API_BASE}/chat",
                        json=payload,
                        headers=_headers(),
                        timeout=TIMEOUT,
                    )
                    body = response.json()
                except httpx.HTTPError:
                    response = None
                    body = {"error": "backend_unavailable"}

            if response is None:
                answer = "后端服务不可用,请先启动 app.main"
            elif response.status_code != 200:
                answer = f"请求失败({response.status_code}):{body.get('detail', body)}"
            elif body.get("interrupted"):
                answer = "该请求已转人工审核,请等待处理结果。"
            elif body.get("final_answer"):
                answer = body["final_answer"]
            else:
                answer = f"出了点问题:{body.get('error') or '未知错误'}"

            st.markdown(answer)

        if response is not None and response.status_code == 200:
            st.session_state.thread_id = body["thread_id"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()


_REVIEW_ACTIONS = {
    "同意(转人工处理)": "accepted_for_manual_processing",
    "驳回": "rejected",
    "需要用户补充信息": "need_more_info",
}


def render_operator() -> None:
    st.title("客服工作台")

    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None

    with st.sidebar:
        user = st.session_state.user
        st.header(f"客服:{user['username']}")
        if st.button("退出登录", use_container_width=True):
            _logout()
            st.rerun()
        if st.button("刷新待审列表", use_container_width=True):
            st.rerun()

    try:
        response = httpx.get(
            f"{API_BASE}/reviews/pending",
            headers=_headers(),
            timeout=TIMEOUT,
        )
        pending = response.json()["pending"] if response.status_code == 200 else []
    except httpx.HTTPError:
        st.error("后端服务不可用,请先启动 app.main")
        pending = []

    if not pending:
        st.info("当前没有待审核的工单。")
        return

    st.subheader(f"待审工单({len(pending)})")
    for task in pending:
        payload = task.get("payload") or {}
        order_id = task.get("order_id") or "未指定订单"
        with st.container(border=True):
            st.markdown(
                f"**{task['task_id']}** | 用户 `{task.get('user_id')}` | "
                f"订单 `{order_id}`"
            )
            st.markdown(f"诉求摘要:{payload.get('reason_summary', '')}")
            st.caption(
                f"thread_id: {task['thread_id']} | 提交时间: {task['created_at']}"
            )

            action = st.radio(
                "处理决定",
                list(_REVIEW_ACTIONS),
                key=f"action_{task['task_id']}",
                horizontal=True,
            )
            note = st.text_input(
                "备注(驳回/需补充时填写,会展示给用户)",
                key=f"note_{task['task_id']}",
            )
            if st.button("提交处理结果", key=f"submit_{task['task_id']}"):
                try:
                    result = httpx.post(
                        f"{API_BASE}/reviews/resume",
                        json={
                            "thread_id": task["thread_id"],
                            "status": _REVIEW_ACTIONS[action],
                            "note": note.strip() or None,
                        },
                        headers=_headers(),
                        timeout=TIMEOUT,
                    )
                    if result.status_code == 200:
                        st.session_state.flash = "已处理,用户侧将看到结果。"
                        st.rerun()
                    else:
                        st.error(
                            f"处理失败({result.status_code}):"
                            f"{result.json().get('detail', result.text)}"
                        )
                except httpx.HTTPError:
                    st.error("后端服务不可用")


if st.session_state.token is None:
    render_auth()
elif st.session_state.user["role"] == "operator":
    render_operator()
else:
    render_customer()
