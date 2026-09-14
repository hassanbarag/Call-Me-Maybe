from pydantic import BaseModel


class FunctionParameter(BaseModel):
    """Represents a single paramter's type within a function definition.."""
    type: str


class FunctionDefinition(BaseModel):
    """Represents one fucnion available for the llm to call."""

    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: dict[str, str] | None = None


class PromptEntry(BaseModel):
    """Represents one natural-language prompt to process."""
    prompt: str


class FunctionCallResult(BaseModel):
    """Represents the final output for
    one prompt: which function, which args."""
    prompt: str
    name: str
    parameters: dict[str, int | float | str | bool]
