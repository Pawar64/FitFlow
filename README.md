# FitFlow
Fitness companion app with workout &amp; diet guidance.
FitFlow is an AI-powered fitness assistant built using **Streamlit**, **OpenAI**, **Google Gemini**, and **Tavily Search**.  
It generates personalized **7-day workout plans**, **7-day Indian diet plans**, includes an **AI chatbot**, and allows exporting plans as a **PDF**.

---

## 📌 Overview

FitFlow helps users create personalized fitness routines using AI. It provides:

- Personalized workout plans  
- Personalized Indian diet plans  
- AI chatbot (FitBuddy)  
- BMI analysis  
- Motivation quotes  
- Web search–based fitness research  

---

## ⭐ Features

### **1. Workout Generator**
AI-generated 7-day workout plan based on:

- Age  
- Height  
- Weight  
- BMI  
- Fitness goal  

### **2. Indian Diet Planner**
Creates a weekly diet plan optimized for Indian food preferences and fitness goals.

### **3. AI Chatbot — FitBuddy**
Ask questions such as:

- “Modify the workout for beginners”  
- “Give vegetarian alternatives”  
- “Is my BMI healthy?”  
- “How much water should I drink?”  

### **4. LangChain Tools Used**
- `tavily_search` — research workouts & diets  
- `analyze_bmi` — explains BMI  
- `motivation_quote` — fitness quote  
- `general_tool` — Gemini-based Q&A  

### **5. PDF Export**
Download both workout & diet plans as a clean PDF.

---

## 🧠 Tech Stack

- Python  
- Streamlit  
- OpenAI GPT-4o Mini  
- Google Gemini 2.5 Flash  
- LangChain  
- Tavily Search API  
- FPDF (PDF generation)  

---

## 📁 Project Structure

FitFlow/
│
├── fitflow_main.py
├── requirements.txt
└── README.md

---  

## 🛠️ Setup Instructions

### **1. Clone the repository**
```bash
git clone https://github.com/Pawar64/FitFlow.git
cd FitFlow
```
### **2. Install dependencies**
```bash
pip install -r requirements.txt
```
### **3. Add API Keys**
Enter these keys inside the Streamlit sidebar:
- OpenAI API Key
- Gemini API Key
- Tavily API Key
  
### **4. Run the app**
```bash
streamlit run fitflow_main.py
```

## 🚀 Usage
- **Workout & Diet Generator**
   -Enter your profile
   - Select your fitness goal
   - Click Generate Plans
   - View plans
   - Export as PDF

- **FitBuddy Chat**
   - Type any fitness-related question
   - AI agent responds with actionable guidance
----

## ❓ Why This Project?

Many people struggle with:
- Understanding what workouts to follow
- Choosing the right diet
- Knowing their BMI
- Staying motivated

-----

## 🙌 Acknowledgments

- OpenAI
- Google Gemini
- Tavily Search
- LangChain
- Streamlit
- FPDF

