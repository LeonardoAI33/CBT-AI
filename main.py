from llama_cpp import Llama
import streamlit as st
import time

# **Carrega o modelo uma única vez**
llm = Llama(
    model_path="/home/andrezib/Desktop/CBT-AI/AI-models/llama-2-7b-chat.Q3_K_M.gguf",
    n_ctx=2048,
    max_tokens=1024,
    temperature=0.7,
    device_map="auto"
)

def generate_response(prompt: str) -> str:
    """Gera resposta via llama-cpp-python."""
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You're a Cognitive Behavioral Therapist specialized in psychology."},
            {"role": "user", "content": prompt}
        ]
    )
    return output['choices'][0]['message']['content']

def typewriter_effect(text: str, speed: float = 0.03):
    """Efeito máquina de escrever."""
    container = st.empty()
    
    displayed = ""
    for ch in text:
        displayed += ch
        container.text(f"\n{displayed}█\n")
        time.sleep(speed)
    container.text(f"\n{displayed}\n")

def main():
    st.title("Julia-AI")

    # **Inicializa histórico**
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # **Renderiza mensagens anteriores**
    for msg in st.session_state.messages:
        role = msg["role"]
        st.chat_message(role).write(msg["content"])

    # **Input do usuário**
    if prompt := st.chat_input("Mensagem:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # **Gera resposta**
        with st.chat_message("ai"):
            with st.spinner("Gerando..."):
                answer = generate_response(prompt)
            typewriter_effect(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()