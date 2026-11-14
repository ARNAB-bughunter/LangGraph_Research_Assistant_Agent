from typing_extensions import TypedDict, NotRequired
from typing import List, Optional
from src.utils.objects import Analyst

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: NotRequired[Optional[str]] # Human feedback
    analysts: NotRequired[List[Analyst]] # Analyst asking questions