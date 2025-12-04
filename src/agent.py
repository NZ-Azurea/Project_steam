import sys
import json
import requests
import re  # <--- IMPORTANT : On importe le module Regex
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple

# --- 1. CONFIGURATION ---
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from library_api_connector import get_unique_tags_categories, search_games_connector
except ImportError as e:
    print(f"Erreur critique d'import : {e}")
    sys.exit(1)

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "openai_gpt-oss-20b-MXFP4.gguf"

# Récupération des données (Limitée)
try:
    _data_info = get_unique_tags_categories()
    TAGS_LIST = ", ".join(_data_info.get("tags", [])[:80]) 
    CATEGORIES_LIST = ", ".join(_data_info.get("categories", []))
except Exception:
    TAGS_LIST = ""
    CATEGORIES_LIST = ""

# --- 2. HELPERS ---

def _format_history(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "No previous conversation."
    formatted = []
    recent_history = history[-4:] 
    for role, content in recent_history:
        role_name = "User" if role == "user" else "Assistant"
        formatted.append(f"{role_name}: {content}")
    return "\n".join(formatted)

def _raw_llm_request(messages: List[Dict[str, str]], stream: bool = True, temperature: float = 0.1, json_mode: bool = False) -> Iterator[str]:
    """Envoie la requête brute."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "n_predict": 1024,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Content-Type": "application/json"}

    try:
        with requests.post(LLAMA_SERVER_URL, headers=headers, json=payload, stream=stream) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line: continue
                if line.startswith(b"data: "): line = line[len(b"data: "):]
                if line == b"[DONE]": break
                try:
                    data = json.loads(line)
                    content = data["choices"][0]["delta"].get("content")
                    if content: yield content
                except json.JSONDecodeError: continue
    except Exception as e:
        yield f"[Error calling LLM: {str(e)}]"

def _filter_thought_process(generator: Iterator[str]) -> Iterator[str]:
    """
    Utilise un REGEX pour trouver la balise de fin de pensée.
    Tant que la balise n'est pas trouvée, on stocke dans le buffer.
    Dès qu'elle est trouvée, on jette le début et on stream le reste.
    """
    buffer = ""
    # Regex qui cherche exactement la balise de séparation
    # On cherche : <|channel|>final<|message|>
    pattern = re.compile(r"<\|channel\|>final<\|message\|>")
    
    found_separator = False

    for chunk in generator:
        if found_separator:
            # CAS 1 : On a déjà trouvé la séparation, on laisse tout passer (mode transparent)
            yield chunk
            continue
        
        # CAS 2 : On cherche encore
        buffer += chunk
        
        # On regarde si notre pattern est DANS le buffer
        match = pattern.search(buffer)
        
        if match:
            # TROUVÉ !
            # match.end() nous donne l'index juste après la balise
            start_of_real_content = match.end()
            
            # On extrait la partie utile (ce qui est après la balise)
            clean_content = buffer[start_of_real_content:]
            
            yield clean_content
            
            # On marque comme trouvé pour ne plus re-tester le regex
            found_separator = True
            buffer = "" # On vide la mémoire

    # CAS DE SECOURS : 
    # Si le stream finit et qu'on a JAMAIS trouvé la balise (le modèle a buggé ou oublié la balise)
    # On renvoie tout le buffer pour ne pas laisser l'utilisateur avec un écran vide.
    if not found_separator and buffer:
        # Petit nettoyage de secours pour enlever au moins le début si possible
        clean_fallback = buffer.replace("<|channel|>analysis<|message|>", "")
        yield clean_fallback

Fonction_Structure="""
    n: int,
    max_price: Optional[float] = None,
    name_contains: Optional[str] = None,
    categories: Optional[List[str]] = None,
    genres: Optional[List[str]] = None,
    min_negative: Optional[int] = None,
    max_negative: Optional[int] = None,
    min_positive: Optional[int] = None,
    max_positive: Optional[int] = None,
    release_date_from: Optional[str] = None,  # ISO string, e.g. "2023-01-01T00:00:00" (le formatage...)
    release_date_to: Optional[str] = None,
    min_required_age: Optional[int] = None,
    max_required_age: Optional[int] = None,
    required_tags: Optional[List[str]] = None,
    verbose: bool = False,
"""
return_structure="""{{"need_function_tag": <bool>, "parameters": <dict of parameters> }}"""

def _get_router_decision(user_message: str, history_str: str) -> Dict[str, Any]:
    """Routeur JSON strict."""
    router_prompt = f"""
    You are a JSON formatting API. You do NOT write code. You do NOT explain. You ONLY output JSON.
    
    AVAILABLE METADATA:
    - Categories: {CATEGORIES_LIST}
    - Tags (partial): {TAGS_LIST}...
    
    TASK:
    Convert the user input into a JSON search query under this format:{return_structure}
    the parameters filed can contain any of the following fields (all optional except n):
    {Fonction_Structure}

    With the following rules:
    required_tags and categories can only use the available metadata listed above.
    when user asks for games that are like other games, please search more game than ask (increase n) because the game compared game might be return by the search engine.
    if the user request is about a previous game, please think about its tags and categories and use them in the search.

    EXAMPLES:
    User: "How much is Borderlands 3?" -> Response: {{ "need_function_tag": true, "parameters": {{ "n": 1, "name_contains": "Borderlands 3" }} }}
    User: "Give me 5 horror games" -> Response: {{ "need_function_tag": true, "parameters": {{ "n": 5, "required_tags": ["Horror"] }} }}
    User: "Hello" -> Response: {{ "need_function_tag": false, "parameters": {{}} }}
    user: "Give me 10 games like phasmophobia" -> Response: {{ "need_function_tag": true, "parameters": {{ "n": 15, "required_tags": ["Horror"] }} }}

    Bad EXAMPLE (DO NOT DO THIS):
    user: "Give me 10 games like phasmophobia" -> Response: {{ "need_function_tag": true, "parameters": {{ "n": 10, "required_tags": ["phasmophobia"] }} }}
    explaination: "phasmophobia" is not a tag, it's a game name.

    ---
    HISTORY: {history_str}
    INPUT: "{user_message}"
    RESPONSE (JSON ONLY):
    """
    
    messages = [{"role": "user", "content": router_prompt}]
    
    full_response = ""
    for chunk in _raw_llm_request(messages, stream=True, temperature=0.0, json_mode=True):
        full_response += chunk
        
    # print(f"\n🧠 [ROUTER RAW]: {full_response}\n") # DEBUG
    
    # Nettoyage
    start = full_response.find("{")
    end = full_response.rfind("}")
    if start != -1 and end != -1:
        clean_text = full_response[start:end+1]
    else:
        clean_text = "{}"

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return {"need_function_tag": False, "parameters": {}}

# --- 3. FONCTION PRINCIPALE ---

def ask_model(message: str, history: List[Tuple[str, str]] = []) -> Iterator[str]:
    import streamlit as st 
    
    history_str = _format_history(history)
    
    # 1. Décision
    decision = _get_router_decision(message, history_str)
    need_function = decision.get("need_function_tag", False)
    params = decision.get("parameters", {})
    context_info = ""

    # 2. Exécution
    if need_function:
        clean_params = {k: v for k, v in params.items() if v not in [None, "", []]}
        if "n" not in clean_params: clean_params["n"] = 5
        
        print(f"🔎 SEARCHING: {clean_params}")
        success, result = search_games_connector(verbose=True, **clean_params)
        
        if success:
            count = result.get("count", 0)
            games_data = result.get("games", [])
            # Sauvegarde Session State
            if "last_search_results" not in st.session_state:
                st.session_state["last_search_results"] = None
            st.session_state["last_search_results"] = games_data
            
            context_info = f"SYSTEM DATABASE RESULTS ({count} found): {json.dumps(games_data, ensure_ascii=False)}"
        else:
            context_info = f"SYSTEM DATABASE ERROR: {result}"
    else:
        context_info = "SYSTEM: No database search needed."

    # 3. Réponse Finale
    final_prompt = f"""
    You are a helpful game assistant.
    INSTRUCTIONS:
    - Use the DATA below to answer.
    - Do NOT write python code.
    - Just write the final answer to the user.
    
    DATA:
    {context_info}
    
    HISTORY:
    {history_str}
    
    USER: {message}
    """

    messages = [
        {"role": "system", "content": final_prompt},
        {"role": "user", "content": message}
    ]

    # Appel au générateur brut
    raw_generator = _raw_llm_request(messages, stream=True, temperature=0.7)
    
    # FILTRAGE REGEX
    for chunk in _filter_thought_process(raw_generator):
        yield chunk