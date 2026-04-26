# 🤖 Multi-Agent System with LangChain + LangGraph

A multi-agent AI system built using **LangChain** and **LangGraph**, featuring orchestrated agent collaboration, shared state passing, and a clearly defined workflow — built as part of a college assignment.

---

## 📌 Overview

This project implements a multi-agent pipeline where specialized agents work together to complete a complex task. Each agent has a distinct role, and LangGraph manages the workflow by defining nodes, edges, and state transitions between them.

---

## ✨ Features

- 🧩 **Multi-Agent Architecture** — 3–4 specialized agents with clear, distinct roles
- 🔀 **LangGraph Workflow** — Nodes and edges define the agent execution flow
- 🔗 **Shared State/Context** — Agents communicate via a shared state object passed through the graph
- 💬 **Dynamic User Input** — Accepts real-time input to drive the pipeline
- 🧠 **LangChain Integration** — Leverages LangChain's LLM and tool abstractions

---

## 🛠️ Tech Stack

| Component         | Technology                  |
|-------------------|-----------------------------|
| Language          | Python                      |
| Agent Framework   | LangChain                   |
| Workflow Engine   | LangGraph                   |
| LLM Backend       | OpenAI GPT / Groq / Ollama  |
| State Management  | LangGraph StateGraph        |

> Update the LLM backend to match what you actually used.

---

## 📁 Project Structure

```
multi-agent-system/
│
├── multi_agent_system.py   # Main file — all agents, graph, and main() function
├── requirements.txt        # Python dependencies
├── .env                    # API keys (do NOT commit this)
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## 🧠 Agent Roles

| Agent         | Role                                              |
|---------------|---------------------------------------------------|
| Agent 1       | _(e.g., Research Agent — gathers relevant info)_  |
| Agent 2       | _(e.g., Planner Agent — structures the output)_   |
| Agent 3       | _(e.g., Writer Agent — generates final content)_  |
| Agent 4       | _(e.g., Reviewer Agent — validates the result)_   |

> Replace the above with your actual agents and their roles.

---

## 🔀 LangGraph Workflow

```
User Input
    │
    ▼
[Agent 1] ──► [Agent 2] ──► [Agent 3] ──► [Agent 4]
                                               │
                                               ▼
                                         Final Output
```

> Update this flow to reflect your actual LangGraph edges.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-system.git
cd multi-agent-system
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the system
```bash
python multi_agent_system.py
```

---

## 🚀 How It Works

1. The user provides input when prompted (e.g., a topic, query, or goal)
2. The **LangGraph StateGraph** kicks off the pipeline
3. Each agent processes the shared state and adds its output
4. The final agent returns the completed result
5. Output is printed to the terminal

---

## 📦 Requirements

See [`requirements.txt`](./requirements.txt). Key packages:

```
langchain
langgraph
langchain-openai
python-dotenv
```

---



> ⭐ Star this repo if you found it helpful!
