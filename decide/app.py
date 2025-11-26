from typing import AsyncIterable, Literal, TypedDict

import logfire
import streamlit as st
from pydantic import BaseModel
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

import pandas as pd

from decide.agent import agent
from decide.storage import BayesianData, DataStore, ModelData, SQLData


class SessionState:
    messages: list[ModelMessage]
    datastore: DataStore

    def __contains__(self, item: str) -> bool:
        raise NotImplementedError("Only for type checking")

SESSION_STATE: SessionState = st.session_state # type: ignore


class ChatMessage(BaseModel):
    """Represents a chat message in the conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


def initialize_session_state():
    """Initialize session state variables for the chat application."""
    if "messages" not in SESSION_STATE:
        SESSION_STATE.messages = []
    if "datastore" not in SESSION_STATE:
        SESSION_STATE.datastore = DataStore()


def to_chat_message(m: ModelMessage) -> ChatMessage | None:
    if len(m.parts) == 0:
        return None
    if isinstance(m, ModelRequest):
        content = ""
        for request_part in m.parts:
            if isinstance(request_part, UserPromptPart) and request_part.content:
                content += str(request_part.content)
        if content:
            return ChatMessage(
                role="user",
                content=content,
            )
        else:
            return None
    elif isinstance(m, ModelResponse):
        content = ""
        for response_part in m.parts:
            if isinstance(response_part, TextPart) and response_part.content:
                content += str(response_part.content)
        if content:
            return ChatMessage(
                role="assistant",
                content=content,
            )
        else:
            return None
    else:
        return None


def display_chat_history():
    """Display the entire chat history using Streamlit chat messages."""
    for message in SESSION_STATE.messages:
        logfire.debug("Raw message: {message}", message=message)
        chat_message = to_chat_message(message)

        if chat_message is None:
            continue

        logfire.debug("Parsed message: {chat_message}", chat_message=chat_message)

        with st.chat_message(chat_message.role):
            st.write(chat_message.content)


def clear_chat_history():
    """Clear the chat history."""
    SESSION_STATE.messages = []
    st.rerun()


def build_upload_file() -> pd.DataFrame | None:
    """Build the file upload interface."""
    st.markdown("Supported file types: CSV, Excel (.xlsx, .xls)")

    if file := st.file_uploader(
        "Choose a file", type=["csv", "xlsx", "xls", "parquet"]
    ):
        try:
            with st.spinner("Loading your data..."):
                match file.type:
                    case "text/csv":
                        df = pd.read_csv(file)
                    case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                        df = pd.read_excel(file)
                    case "application/octet-stream":
                        df = pd.read_parquet(file)
                    case _:
                        st.error(f"Unsupported file type {file.type}")
                        return None
                df = df.infer_objects()
                st.success(
                    f"✅ Successfully loaded **{len(df)} rows** and **{len(df.columns)} columns**"
                )
                return df

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info(
                "💡 Make sure your file is not corrupted and is in the correct format."
            )
            return None

    return None


def build_data_viewer() -> None:
    """Build the data viewer interface."""
    st.header("📊 Data Viewer")

    names = SESSION_STATE.datastore.keys()
    tabs = st.tabs(names)
    for name, tab in zip(names, tabs):
        data = SESSION_STATE.datastore.get(name)
        with tab:
            st.write(data.description)
            match data:
                case SQLData():
                    st.write("Inputs: " + ", ".join(data.input_data))
                    with st.expander("🔍 View SQL"):
                        st.code(data.sql, language="sql")
                case ModelData():
                    st.write("Input: " + data.input_data)
                    with st.expander("🔍 View Model"):
                        st.code(str(data.model))
                case BayesianData():
                    st.write("Input: " + data.input_data)
                    with st.expander("🔍 View Model Code"):
                        st.code(data.model_code, language="python")
                    with st.expander("📊 View Posterior Summary"):
                        st.dataframe(data.summary)

            st.dataframe(data.df, width="stretch")


async def handle_events(_, event_stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in event_stream:
        logfire.debug("Event: {event}", event=event)
        match event:
            case FunctionToolCallEvent():
                st.write(f"Calling {event.part.tool_name} with args:")
                st.code(event.part.args)
            case FunctionToolResultEvent():
                st.write(f"Tool {event.result.tool_name} returned:")
                st.code(event.result.content)


def build_chat() -> None:
    """Build the chat interface with the data analysis agent."""

    st.header("💬 Chat with Your Data")

    if st.button("🗑️ Clear Chat", help="Clear conversation history"):
        clear_chat_history()

    display_chat_history()

    if prompt := st.chat_input("Ask me about your data..."):
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                with st.status("🤔 Analyzing your data..."):
                    response = agent.run_sync(
                        user_prompt=prompt,
                        deps=SESSION_STATE.datastore,
                        message_history=SESSION_STATE.messages,
                        event_stream_handler=handle_events,
                    )
            except Exception as e:
                error_message = (
                    f"An error occurred while processing your request: {str(e)}"
                )
                st.error(error_message)
                st.exception(e)
            else:
                SESSION_STATE.messages.extend(response.new_messages())
                for message in response.new_messages():
                    logfire.debug("Message: {message}", message=message)
                    chat_message = to_chat_message(message)
                    if chat_message is None:
                        continue
                    logfire.debug(
                        "Parsed message: {chat_message}", chat_message=chat_message
                    )
                    st.write(chat_message.content)


def build_page() -> None:
    """Build the main page of the application."""
    st.set_page_config(
        page_title="Decide - Data Analysis Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    initialize_session_state()

    st.title("📊 Decide - Data Analysis Assistant")
    st.header("📁 Data Upload")

    data = build_upload_file()
    if data is not None:
        if "initial" not in SESSION_STATE.datastore:
            SESSION_STATE.datastore.store_static(
                df=data,
                name="initial",
                description="The initial dataframe, uploaded by the user.",
            )
        container = st.container()
        build_chat()
        with container:
            build_data_viewer()
