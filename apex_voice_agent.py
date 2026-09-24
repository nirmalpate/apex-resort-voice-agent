import os
import uuid
import snowflake.connector
from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

# 1. Database Tool
def get_snowflake_connection():
    return snowflake.connector.connect(
        user=os.environ.get("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        account=os.environ.get("SNOWFLAKE_ACCOUNT"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database="HOTEL_DB",
        schema="RECEPTION"
    )

@tool
def query_room_inventory(room_category: str = "all") -> str:
    """Queries Snowflake HOTEL_DB for available rooms, nightly rates, and capacity limits."""
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        if room_category.lower() == "all":
            cursor.execute("SELECT ROOM_TYPE, PRICE_PER_NIGHT, MAX_GUESTS FROM ROOM_INVENTORY WHERE IS_AVAILABLE = TRUE;")
        else:
            cursor.execute(
                "SELECT ROOM_TYPE, PRICE_PER_NIGHT, MAX_GUESTS FROM ROOM_INVENTORY WHERE IS_AVAILABLE = TRUE AND LOWER(ROOM_TYPE) LIKE %s;",
                (f"%{room_category.lower()}%",)
            )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not rows:
            return f"No available rooms found matching category: {room_category}"
            
        summary = ["Available rooms at Apex Resort:"]
        for row in rows:
            summary.append(f"- {row[0]}: ${row[1]}/night (Capacity: {row[2]} guests)")
        return "\n".join(summary)
    except Exception as err:
        return f"Error querying Snowflake database: {str(err)}"

tools = [query_room_inventory]
tools_by_name = {t.name: t for t in tools}

# 2. Schemas & State
class ReservationDetails(BaseModel):
    guest_name: Optional[str] = Field(default=None, description="Full name of guest")
    check_in_date: Optional[str] = Field(default=None, description="Check-in date")
    check_out_date: Optional[str] = Field(default=None, description="Check-out date")
    requested_services: Optional[List[str]] = Field(default=None, description="Requested resort services")

class ReceptionistState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    guest_name: Optional[str]
    check_in_date: Optional[str]
    check_out_date: Optional[str]
    requested_services: Optional[List[str]]

# 3. Model Setup
llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)
extractor = llm.with_structured_output(ReservationDetails)

# 4. Graph Logic
def parse_guest_info_node(state: ReceptionistState):
    updates = {}
    try:
        extracted: ReservationDetails = extractor.invoke(state["messages"])
        if extracted:
            if extracted.guest_name: updates["guest_name"] = extracted.guest_name
            if extracted.check_in_date: updates["check_in_date"] = extracted.check_in_date
            if extracted.check_out_date: updates["check_out_date"] = extracted.check_out_date
            if extracted.requested_services and isinstance(extracted.requested_services, list):
                existing = state.get("requested_services") or []
                updates["requested_services"] = list(set(existing + extracted.requested_services))
    except Exception as e:
        print(f"[PARSER WARNING] Structured extraction fallback: {e}")
    return updates

def agent_dialogue_node(state: ReceptionistState):
    known_name = state.get("guest_name", "Not provided")
    known_checkin = state.get("check_in_date", "Not provided")
    known_checkout = state.get("check_out_date", "Not provided")
    known_services = state.get("requested_services", [])

    system_prompt = SystemMessage(
        content=(
            "You are Maya, senior AI receptionist at Apex Luxury Resort.\n"
            "CURRENT BOOKING CONTEXT:\n"
            f"- Guest Name: {known_name}\n"
            f"- Check-In: {known_checkin}\n"
            f"- Check-Out: {known_checkout}\n"
            f"- Requested Services: {', '.join(known_services) if known_services else 'None'}\n\n"
            "INSTRUCTIONS:\n"
            "1. Call `query_room_inventory` for room rates or availability.\n"
            "2. Respond concisely, warmly, and ask for missing reservation details naturally."
        )
    )
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def execute_tools_node(state: ReceptionistState):
    last_msg = state["messages"][-1]
    tool_results = []
    for call in last_msg.tool_calls:
        selected_tool = tools_by_name[call["name"]]
        output = selected_tool.invoke(call["args"])
        tool_results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": tool_results}

def determine_next_step(state: ReceptionistState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

memory = MemorySaver()
graph_builder = StateGraph(ReceptionistState)

graph_builder.add_node("parser", parse_guest_info_node)
graph_builder.add_node("agent", agent_dialogue_node)
graph_builder.add_node("tools", execute_tools_node)

graph_builder.add_edge(START, "parser")
graph_builder.add_edge("parser", "agent")
graph_builder.add_conditional_edges("agent", determine_next_step, ["tools", END])
graph_builder.add_edge("tools", "agent")

receptionist_app = graph_builder.compile(checkpointer=memory)
