import operator
from typing import List, Optional

from langgraph.graph import MessagesState
from typing_extensions import Annotated, NotRequired, TypedDict

from src.utils.objects import Analyst


class GenerateAnalystsState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: NotRequired[Optional[str]]  # Human feedback
    analysts: NotRequired[List[Analyst]]  # Analyst asking questions


class InterviewState(MessagesState):
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    analyst: Analyst  # Analyst asking questions
    interview: str  # Interview transcript
    sections: list  # Final key we duplicate in outer state for Send() API
