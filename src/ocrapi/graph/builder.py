from functools import lru_cache, partial

from langgraph.graph import END, START, StateGraph

from ocrapi.graph.nodes import aggregate_node, ingest_node, ocr_node
from ocrapi.graph.state import OCRState
from ocrapi.ocr.sglang_client import SGLangOCRClient


def build_graph(client: SGLangOCRClient | None = None):
    client = client or SGLangOCRClient()

    builder = StateGraph(OCRState)
    builder.add_node("ingest", ingest_node)
    builder.add_node("ocr", partial(ocr_node, client=client))
    builder.add_node("aggregate", aggregate_node)
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "ocr")
    builder.add_edge("ocr", "aggregate")
    builder.add_edge("aggregate", END)

    return builder.compile()


@lru_cache
def get_compiled_graph():
    """Process-wide singleton used as a FastAPI dependency.

    Overridden in route tests via app.dependency_overrides; graph logic
    itself is tested directly via build_graph(client=FakeSGLangClient()).
    """
    return build_graph()
