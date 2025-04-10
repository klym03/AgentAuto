import os
from telethon import TelegramClient, events
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from vector_store import create_vector_store
from dotenv import load_dotenv

load_dotenv()
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE")

client = TelegramClient("session_name", API_ID, API_HASH)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    convert_system_message_to_human=True
)

vector_store = create_vector_store("data/car_info.txt")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Пам'ять, що зберігає історію чату

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)

@client.on(events.NewMessage(incoming=True))
async def handle_message(event):
    sender = await event.get_sender()
    user_id = sender.id
    message_text = event.message.text

    me = await client.get_me()
    if user_id == me.id:
        return

    response = qa_chain.invoke({"question": message_text})["answer"]

    await event.reply(response)


async def main():
    await client.start(phone=PHONE)
    print("Агент AutoDream запущений. Чекаю повідомлень...")
    await client.run_until_disconnected()


if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(main())
