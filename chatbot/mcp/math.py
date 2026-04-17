from fastmcp import FastMCP

mcp = FastMCP("math-mcp-server")

@mcp.tool
def add(a: int, b: int) -> int:
    """ Adds two numbers """
    return a + b

@mcp.tool
def sub(a: int, b: int) -> int:
    """ Subtracts two numbers """
    return a - b

@mcp.tool
def mul(a: int, b: int) -> int:
    """ Multiplies two numbers """
    return a * b

@mcp.tool
def div(a: int, b: int) -> int:
    """ Divides two numbers """
    return a / b

if __name__ == "__main__":
    mcp.run(transport="stdio")