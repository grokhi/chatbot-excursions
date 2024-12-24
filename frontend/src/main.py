import os
import uuid

import gradio as gr
import requests
import requests.cookies

API_BASE_URL = "http://{}:{}".format(os.getenv("HOST", "localhost"), os.getenv("PORT", 8000))


def query_chatbot(chat_history, user_message, state):
    """
    Send a query to the backend and return the updated chat history.

    Args:
        chat_history (list): Current chat history as a list of messages.
        user_message (str): User's message to the chatbot.
        state (dict): Conversation's dict with unique session id.

    Returns:
        list: Updated chat history with the bot's response.
    """
    # Ensure the chat_history is a list of dictionaries
    if not chat_history:
        chat_history = []

    if "session_id" not in state:
        state["session_id"] = str(uuid.uuid4())

    _cookie_jar = requests.cookies.RequestsCookieJar()
    _cookie_jar.set("session", state["session_id"])

    chat_history.append({"role": "user", "content": user_message})

    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": user_message},
            cookies=_cookie_jar,
            allow_redirects=True,
        )
        if response.status_code == 200:
            data = response.json()
            chat_history.append({"role": "assistant", "content": data["answer"]})
        else:
            error_message = f"Error: {response.status_code}\n{response.text}"
            chat_history.append({"role": "assistant", "content": error_message})
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})

    return chat_history, ""


# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# 🤖 RAG Chatbot")

    state = gr.State({})

    with gr.Column():
        chat_display = gr.Chatbot(label="Chat with RAG Bot", type="messages")
        user_input = gr.Textbox(label="", placeholder="Type your message here...")
        send_button = gr.Button("Send")

    # Event to update the chat history
    send_button.click(
        query_chatbot,
        inputs=[chat_display, user_input, state],
        outputs=[chat_display, user_input],
        queue=False,
    )

# Launch the app
if __name__ == "__main__":
    app.launch()
