┌─────────────────────────────────────────────────────────────┐
│                  🧠 User Query Input (Frontend / CLI)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    🌐 LangChain Orchestrator                │
│  (LLMChain / Agents / Tools / Routing Logic)                │
└─────────────────────────────────────────────────────────────┘
           │                 │                  │
           ▼                 ▼                  ▼
 ┌────────────────┐  ┌────────────────┐  ┌─────────────────────┐
 │ 🛠 Tool Agent   │  │ 📊 Search Agent │  │ 💻 Code Agent (E2B) │
 │ - Uses Python  │  │ - Uses Web     │  │ - Uses e2b sandbox  │
 │ - Uses REPL    │  │   search tool  │  │   to run code safely│
 └────────────────┘  └────────────────┘  └─────────────────────┘
           ▼                 ▼                  ▼
                      🧠 Agent Outputs / Sub-Decisions
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │      🔁 LangGraph (Multi-Agent Feedback Loop) │
        │  (Optional - to chain agent responses in logic) │
        └────────────────────────────────────────────┘
                              │
                              ▼
      ┌──────────────────────────────────────────────┐
      │       🧾 Final Output / Suggested Response     │
      │  (to frontend / dashboard / chat UI / API)     │
      └──────────────────────────────────────────────┘
One sandbox to execute all the tools
Gradio UI