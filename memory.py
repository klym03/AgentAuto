from langchain.memory import ConversationBufferMemory


class ConversationMemory:
    def __init__(self, buffer_size=5):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_buffer_size=buffer_size
        )

    def __call__(self, inputs):
        return self.memory