import streamlit as st

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")

# ---- MASQUER LA SIDEBAR ----
hide_sidebar_style = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebar"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>Page du Chatbot</h1>", unsafe_allow_html=True)

if st.button("🏠 Accueil"):
    st.switch_page("app.py")

# ---- Exemple : fonction LLM ----
# Remplace ça par ton import réel
def agent_ai_anwser(msg):
    return f"🤖 Réponse du LLM : {msg[::-1]}"  # FAKE → remplace par ta vraie fonction


# ---- INITIALISER L'HISTORIQUE ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---- FONCTION : ENVOYER UN MESSAGE ----
def send_message():
    user_msg = st.session_state.user_input.strip()
    if user_msg == "":
        return
    
    # Ajouter message utilisateur
    st.session_state.chat_history.append(("user", user_msg))

    # Appeler ton agent
    try:
        bot_reply = agent_ai_anwser(user_msg)
    except Exception as e:
        bot_reply = f"Erreur : {e}"

    # Ajouter réponse bot
    st.session_state.chat_history.append(("bot", bot_reply))

    # reset champ input
    st.session_state.user_input = ""


# ---- AFFICHAGE DU CHAT ----
st.write("----")

for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(
            f"""
            <div style="
                background-color:#DCF8C6;
                padding:12px;
                border-radius:10px;
                margin:8px 0;
                max-width:60%;
                margin-left:auto;
                text-align:right;
                ">
                <b>👤 Vous :</b><br>{msg}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div style="
                background-color:#ECECEC;
                padding:12px;
                border-radius:10px;
                margin:8px 0;
                max-width:60%;
                text-align:left;
                ">
                <b>🤖 Bot :</b><br>{msg}
            </div>
            """, unsafe_allow_html=True)


# ---- INPUT UTILISATEUR ----
st.text_input(
    "Votre message :",
    key="user_input",
    on_change=send_message,
    placeholder="Écrivez un message et appuyez sur Entrée..."
)
