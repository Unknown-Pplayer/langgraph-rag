from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool

# web_search_tool = TavilySearchResults(k=3)

# search = DuckDuckGoSearchAPIWrapper()
# web_search_tool = Tool(
#     name="Search",
#     func=search.run,
#     description="Useful for when you need to answer questions about current events. You should ask targeted questions"
# )

search = DuckDuckGoSearchAPIWrapper()

# Create a custom function to use results instead of run


def search_with_results(query: str) -> list:
    return search.results(query, max_results=3)


# Create the tool with the custom function
web_search_tool = Tool(
    name="Search",
    func=search_with_results,
    description="Useful for when you need to answer questions about current events. You should ask targeted questions"
)
