import os
from dotenv import load_dotenv
load_dotenv()

from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from langgraph.store.memory import InMemoryStore
from langchain_openai import ChatOpenAI


# Create model
model = ChatOpenAI(temperature=0)

CLINIC_NAME = "Good Health Clinic"

# This message will provide context to the LLM about its role and the "patient" data it should use.
MODEL_SYSTEM_MESSAGE = """You are a helpful medical assistant for {clinic_name}. 
Use the patient's history to provide relevant, personalized appointment scheduling or advice.
Patient profile: {history}"""

# Instruction for how we update the patient profile (storing appointment data, medical notes, etc.).
UPDATE_PATIENT_PROFILE_INSTRUCTION = """Update the patient's medical/appointment profile with new information.

CURRENT PROFILE:
{history}

ANALYZE FOR:
1. Appointment history (dates, times, no-shows)
2. Medical preferences or concerns
3. Previous diagnoses or treatments
4. Medication usage or allergies
5. Follow-up needs

Focus on verified appointment and medical details only. Summarize key points clearly.

Update the profile based on this conversation:
"""

def check_condition(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    A simple routing node that checks if the last user message
    contains the word 'emergency'. If so, we return
    {'decision': 'emergency_route'}, else {'decision': 'regular_route'}.
    """
    user_msg = state["messages"][-1].content.lower()
    if "emergency" in user_msg:
        return {'decision': 'emergency_route'}
    else:
        return {'decision': 'regular_route'}

def handle_emergency(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    A specialized function that might provide urgent instructions or 
    escalate the flow for an 'emergency' scenario.
    """
    # We could use a model or just return a static response.
    return {
        "messages": [
            SystemMessage(
                content="We’ve detected an emergency. Please contact emergency services immediately or call our 24/7 urgent line: +43 00 00 00."
            )
        ]
    }

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Generates an AI response, leveraging the patient's history for context.
    
    Args:
        state (MessagesState): Current conversation messages
        config (RunnableConfig): Runtime configuration with patient_id
        store (BaseStore): Persistent storage for patient data
        
    Returns:
        dict: Generated response messages
    """
    # 1. Retrieve patient ID and profile from store
    patient_id = config["configurable"]["patient_id"]
    namespace = ("patient_interactions", patient_id)
    key = "patient_data_memory"
    
    memory = store.get(namespace, key)
    
    # 2. Extract existing history or set a default
    history = memory.value.get('patient_data_memory') if memory else "No existing patient profile found."
    
    # 3. Format the system message with the patient's context
    system_msg = MODEL_SYSTEM_MESSAGE.format(history=history, clinic_name=CLINIC_NAME)
    
    # 4. Generate the AI response
    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])
    
    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Updates the patient's appointment/medical profile in persistent storage.
    
    Args:
        state (MessagesState): Current conversation messages
        config (RunnableConfig): Runtime config containing patient_id 
        store (BaseStore): Persistent storage for patient data
    """
    # 1. Retrieve patient history
    patient_id = config["configurable"]["patient_id"]
    namespace = ("patient_interactions", patient_id)
    key = "patient_data_memory"
    
    memory = store.get(namespace=namespace, key=key)
    
    # 2. Extract existing profile or set a default
    history = memory.value.get(key) if memory else "No existing history."
    
    # 3. Generate updated profile content based on the new conversation
    system_msg = UPDATE_PATIENT_PROFILE_INSTRUCTION.format(history=history)
    new_insights = model.invoke([SystemMessage(content=system_msg)] + state['messages'])
    
    # 4. Store updated profile
    #    Here we save the updated profile text under 'patient_data_memory'
    store.put(namespace, key, {"patient_data_memory": new_insights.content})

# Build the graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("check_condition", check_condition)
builder.add_node("call_model", call_model)
builder.add_node("handle_emergency", handle_emergency)
builder.add_node("write_memory", write_memory)

# Add initial edge from START -> check_condition
builder.add_edge(START, "check_condition")
# builder.set_entry_point("check_condition")

# Use add_conditional_edges to branch:
#   If 'decision' == 'emergency_route', go to 'handle_emergency'
#   If 'decision' == 'regular_route', go to 'call_model'
builder.add_conditional_edges(
    "check_condition",
    lambda state: state["decision"],
    {
        "emergency_route": "handle_emergency",
        "regular_route": "call_model",
        "end": END
    }
)

# After either call_model or handle_emergency, go to write_memory
builder.add_edge("handle_emergency", "write_memory")
builder.add_edge("call_model", "write_memory")

# Then from write_memory -> END
builder.add_edge("write_memory", END)

# Initialize memory stores
across_thread_memory = InMemoryStore()   # Long-term storage for patient interactions
within_thread_memory = MemorySaver()     # Keeps current conversation state

# Compile the graph with memory configuration
graph = builder.compile(
    checkpointer=within_thread_memory,  # Tracks conversation state in memory
    store=across_thread_memory          # Persists patient data
)

# Optionally, visualize the graph
# display(Image(graph.get_graph(xray=1).draw_png()))
png_graph = graph.get_graph().draw_mermaid_png()
with open("my_graph.png", "wb") as f:
    f.write(png_graph)

# Configuration for the "patient"
config = {
    "configurable": {
        "thread_id": "1",      # Current conversation ID
        "patient_id": "1"      # Identify the patient in the store
    }
}

# Example patient message
input_msg = [
    HumanMessage(content="Hi, I'm Taher. I'd like to schedule an appointment for a routine check-up.")
]

# Stream the result of the conversation
for chunk in graph.stream(
    {"messages": input_msg},  # Current user message
    config,                   # Our runtime configuration
    stream_mode="values"      # We just want the messages content
):
    chunk["messages"][-1].pretty_print()
    

# print("\n---- EMERGENCY CASE ----")
# emergency_msg = [
#     HumanMessage(content="This is an emergency! I'm experiencing severe chest pain.")
# ]
# for chunk in graph.stream({"messages": emergency_msg}, config, stream_mode="values"):
#     chunk["messages"][-1].pretty_print()
    
# Now Taher replies with a date/time preference:
patient_followup_msg = [
    HumanMessage(
        content="Yes, I'd like to schedule it for next Tuesday around 10 AM, if possible."
    )
]

for chunk in graph.stream(
    {"messages": patient_followup_msg},  # Next user message
    config,                              # Same config with patient_id="1"
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
    
# Patient (Taher) responds, concluding the conversation and requesting a summary:
final_user_msg = [
    HumanMessage(
        content="No, that's all for now. Could you give me a brief summary of my appointment details?"
    )
]

# Send this final message through the same graph:
for chunk in graph.stream(
    {"messages": final_user_msg},  # Next user message
    config,                        # Same runtime config { "configurable": { "patient_id":"1", ...}}
    stream_mode="values"
):
    # The AI's final response (including summary) is in chunk["messages"][-1]
    chunk["messages"][-1].pretty_print()

# OPTIONAL: If you want to retrieve the entire "patient profile" from memory and display it:
# namespace = ("patient_interactions", "1")  # patient_id = "1"
# key = "patient_data_memory"
# memory_data = across_thread_memory.get(namespace, key)

# if memory_data:
#     patient_profile = memory_data.value.get("patient_data_memory")
#     print("\n--- STORED PATIENT PROFILE ---")
#     print(patient_profile)