from llama_cpp import Llama
import streamlit as st
import time


llm = Llama(
    model_path="/home/andrezib/Desktop/CBT-AI/AI-models/llama-2-7b-chat.Q3_K_M.gguf",
    n_ctx=2048,
    max_tokens=1024,
    temperature=0.7,
    device_map="auto"
)

def generate_response(prompt):
    output = llm.create_chat_completion(
    messages = [{"role":"user", "content": prompt}])
    response = output['choices'][0]['message']["content"]
    return response

def typewritter_effect(text, speed=0.03):
    container = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        container.markdown(f"```\n{displayed_text}█\n```")
        time.sleep(speed)
    container.markdown(f"```\n{displayed_text}\n```")
   

def main():
    st.title("CBT-AI")
    st.write("Start your Cognitive Behavioral Therapy Session")
    user_input = st.text_input("Mensagem:")
    if st.button("Send"):
     with st.spinner("In progress..."):
        time.sleep(2)
        response = generate_response(user_input)
        typewritter_effect(response)

if __name__ == "__main__":
   main()