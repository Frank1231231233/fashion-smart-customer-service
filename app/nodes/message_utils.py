from langchain_core.messages import AnyMessage, HumanMessage


def latest_human_context(
    messages: list[AnyMessage],
    current_query: str,
) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        if not isinstance(message.content, str):
            continue

        content = message.content.strip()
        if content and content != current_query:
            return content

    return None


def history_before_current_message(
    messages: list[AnyMessage],
    current_query: str,
) -> list[AnyMessage]:
    if not messages:
        return []

    last_message = messages[-1]
    if (
        isinstance(last_message, HumanMessage)
        and isinstance(last_message.content, str)
        and last_message.content.strip() == current_query.strip()
    ):
        return messages[:-1]

    return messages
