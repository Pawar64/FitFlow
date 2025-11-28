import streamlit as st
import os
from tavily import TavilyClient
from fpdf import FPDF
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------ APP TITLE ------------------
st.markdown("""
    <h1 style='text-align: center; color:#FFBF00; font-size: 50px;'>
        🏋️‍♂️ FitFlow — Your AI Fitness Companion
    </h1>
""", unsafe_allow_html=True)


# ------------------ SIDEBAR: API KEYS ------------------
st.sidebar.header("🔑 API Keys")
openai_api = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_api = st.sidebar.text_input("Gemini API Key", type="password")
tavily_api = st.sidebar.text_input("Tavily API Key", type="password")

if openai_api:
    os.environ["OPENAI_API_KEY"] = openai_api
if gemini_api:
    os.environ["GOOGLE_API_KEY"] = gemini_api
if tavily_api:
    os.environ["TAVILY_API_KEY"] = tavily_api
if not openai_api:
    st.warning("⚠️ Please enter your *OpenAI API Key* in the sidebar.")
    st.stop()
if not gemini_api:
    st.warning("⚠️ Please enter your *Gemini API Key* in the sidebar.")
    st.stop()
if not tavily_api:
    st.warning("⚠️ Please enter your *Tavily API Key* in the sidebar.")
    st.stop()


# ------------------ PROFILE CREATION ------------------
st.sidebar.header("👤 Profile")
if "profile" not in st.session_state:
    st.session_state["profile"] = {}
if "editing_profile" not in st.session_state:
    st.session_state["editing_profile"] = True

profile = st.session_state["profile"]

if st.session_state["editing_profile"]:
    st.sidebar.subheader("Create / Edit Profile")
    name = st.sidebar.text_input("Name", value=profile.get("name", ""))
    age = st.sidebar.number_input("Age", 10, 100, value=int(profile.get("age", 23)))
    height_m = st.sidebar.number_input("Height (m)", 0.8, 2.5, value=float(profile.get("height_m", 1.6)))
    weight_kg = st.sidebar.number_input("Weight (kg)", 20.0, 300.0, value=float(profile.get("weight_kg", 65.0)))

    if st.sidebar.button("Save Profile"):
        bmi = round(weight_kg / (height_m ** 2), 2)
        st.session_state["profile"] = {
            "name": name,
            "age": age,
            "height_m": height_m,
            "weight_kg": weight_kg,
            "bmi": bmi
        }
        st.session_state["editing_profile"] = False
        st.sidebar.success(f"Profile saved! BMI = {bmi}")
        st.rerun()
else:
    st.sidebar.subheader("Your Profile")
    st.sidebar.write(f"*Name:* {profile['name']}")
    st.sidebar.write(f"*Age:* {profile['age']}")
    st.sidebar.write(f"*Height:* {profile['height_m']} m")
    st.sidebar.write(f"*Weight:* {profile['weight_kg']} kg")
    st.sidebar.write(f"*BMI:* {profile['bmi']}")
    if st.sidebar.button("Edit Profile"):
        st.session_state["editing_profile"] = True
        st.rerun()


# ------------------ GOAL SELECTION ------------------
st.sidebar.header("🎯 Fitness Goal")
st.session_state["goal"] = st.sidebar.selectbox(
    "Select your goal", ["Lose Weight", "Gain Muscle", "Maintain", "Tone/Lean"]
)


# ------------------ TAVILY CLIENT ------------------
tavily_client = TavilyClient(api_key=tavily_api)

# ------------------ LLM ------------------
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api,
    temperature=0.2
)

# ------------------TOOLS AND AGENT--------------------- 
from langchain.tools import tool
import random

# GENERAL TOOL
@tool
def general_tool(query: str) -> str:
    """Answer any general (non-fitness) question using Gemini."""
    g_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api,
        temperature=0.2
    )
    return g_llm.invoke(query).content

# MOTIVATION QUOTE TOOL
@tool
def motivation_quote(input: str) -> str:
    """Return a random motivational fitness quote."""
    quotes = [
        "Every workout counts. Keep going! 💪",
        "Small steps every day lead to big changes.",
        "You don’t have to be extreme, just consistent.",
        "Your only limit is your mind. You got this! 🔥"
    ]
    return random.choice(quotes)

# TAVILY SEARCH TOOL
@tool
def tavily_search(q: str) -> str:
    """Search the internet using Tavily and return summarized results."""
    try:
        r = tavily_client.search(query=q)
        results = [x.get("content", "") for x in r.get("results", [])]
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Tavily API Error: {str(e)}"

# BMI ANALYZER TOOL
@tool
def analyze_bmi(bmi: str) -> str:
    """Analyze BMI value and return fitness advice."""
    
    bmi = float(bmi)  # convert input string to number
    
    if bmi < 18.5:
        return "You are underweight. Increase calories and add strength training."
    elif 18.5 <= bmi <= 24.9:
        return "Your BMI is normal. Maintain with balanced training & nutrition."
    elif 25 <= bmi <= 29.9:
        return "You are overweight. Focus on fat-loss, cardio, and a calorie deficit."
    else:
        return "You are in the obesity range. Prioritize walking + low-impact workouts."


# ------------------ NEW AGENT  ------------------
from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType

tools = [
    tavily_search,
    analyze_bmi,                    
    motivation_quote,
    general_tool    
]
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


# ------------------ OPENAI HELPER FOR PLANS ------------------
client = OpenAI(api_key=openai_api)

def generate_openai_response(prompt, model="gpt-4o-mini", temperature=0.3, max_tokens=1500):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful fitness assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API Error: {str(e)}"
    
    
# ------------------ AI WORKOUT GENERATOR ------------------
def generate_workout_with_ai(profile, goal):
    research = tavily_search(
        f"Best 7-day home-friendly workout plan for {goal} for a {profile['age']} year old "
        f"{profile['weight_kg']} kg {profile['height_m']} m person"
    )
    
    if research == "No results found.":
        research = "Use your knowledge to create a home-friendly bodyweight-only plan."

    prompt = f"""
You are a professional fitness trainer. Create a 7-day home-friendly workout plan.
{research}
"""
    return generate_openai_response(prompt)


# ------------------ AI DIET GENERATOR ------------------
def generate_diet_with_ai(profile, goal):
    research = tavily_search(
        f"Best meal plan for {goal} with calories and macros for a {profile['weight_kg']} kg person. Include Indian meals if possible."
    )

    if research == "No results found.":
        research = "Use your knowledge to create the diet plan."

    prompt = f"""
You are a certified nutritionist. Create a 7-day Indian diet plan.
{research}
"""
    return generate_openai_response(prompt)


# ------------------ PLAN GENERATION ------------------
st.sidebar.header("🏋️ Workout & Diet Plans")
if "workout_plan" not in st.session_state:
    st.session_state["workout_plan"] = ""
if "diet_plan" not in st.session_state:
    st.session_state["diet_plan"] = ""

if st.sidebar.button("Generate Plans"):
    if "bmi" not in profile:
        st.sidebar.warning("⚠️ Please complete your profile first.")
    else:
        with st.spinner("Generating AI-powered workout & diet plans..."):
            st.session_state["workout_plan"] = generate_workout_with_ai(profile, st.session_state["goal"])
            st.session_state["diet_plan"] = generate_diet_with_ai(profile, st.session_state["goal"])
        st.sidebar.success("AI Plans Generated! Scroll down.")


# ------------------ SHOW RESULTS ------------------
if st.session_state["workout_plan"] or st.session_state["diet_plan"]:
    tab1, tab2 = st.tabs(["🏋️ Workout Plan", "🍽️ Diet Plan"])
    with tab1:
        st.markdown("## 🏋️ Your 7-Day Home Workout Plan")
        st.markdown(st.session_state["workout_plan"])
    with tab2:
        st.markdown("## 🍽️ Your Personalized Diet Plan")
        st.markdown(st.session_state["diet_plan"])


# ------------------ CHATBOT USING NEW AGENT ------------------
st.header("💬 FitBUDDY")
if "history" not in st.session_state:
    st.session_state["history"] = []

def agent_chat(query):
    context = f"""
USER PROFILE:
{profile}
WORKOUT PLAN:
{st.session_state.get("workout_plan", "")}
DIET PLAN:
{st.session_state.get("diet_plan", "")}
USER QUESTION:
{query}
"""
    try:
        result = agent_executor.invoke({"input": context})
        return result.get("output", result.get("result", "No output"))  
    except Exception as e:
        return f"Agent Error: {str(e)}"

query = st.text_input("Ask anything 👇")
if query:
    reply = agent_chat(query)
    st.session_state["history"].append({"query": query, "response": reply})
    st.write(reply)


# ------------------ EXPORT PDF ------------------
st.sidebar.header("📄 Export Plans")
def clean_text(text):
    if not text:
        return ""
    replacements = {
        "🏋️‍♂️": "[Workout]",
        "🍽️": "[Diet]",
        "–": "-",   # en dash
        "—": "-",   # em dash
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "-",
        "…": "...",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

if st.sidebar.button("Download Plans as PDF"):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Workout & Diet Plans - {profile.get('name', '')}", ln=True)

    # Body
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.multi_cell(0, 10, clean_text("Workout Plan:\n" + st.session_state.get("workout_plan", "")))
    pdf.ln(5)
    pdf.multi_cell(0, 10, clean_text("Diet Plan:\n" + st.session_state.get("diet_plan", "")))

    # Output PDF
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    st.sidebar.download_button(
        label="📄 Download Plans as PDF",
        data=pdf_bytes,
        file_name="plans.pdf",
        mime="application/pdf"
    )


