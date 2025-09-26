import logging
from pathlib import Path
from strands import Agent, tool
from strands.models import BedrockModel
from strands.multiagent.a2a import A2AServer

logging.getLogger("strands").setLevel(logging.INFO)

from dotenv import load_dotenv

load_dotenv(override=True)  # take environment variables

# Constants
MARKDOWN_EXTENSIONS = ['.md', '.markdown']
FILES_DIR = "files"


@tool
def read_markdown_file(file_name: str) -> str:
    """Read the content of a Markdown file from the files directory.
    Args:
        file_name (str): The name of the Markdown file to read (without path).
    Returns:
        str: The content of the Markdown file or error message.
    """
    try:
        # Always work in the files directory
        files_dir = Path(FILES_DIR)
        file_path = files_dir / file_name
        
        # Ensure .md extension if not provided
        if file_path.suffix.lower() not in MARKDOWN_EXTENSIONS:
            file_path = file_path.with_suffix('.md')
        
        if not file_path.exists():
            return (f"Error: File '{file_path}' does not exist in "
                    f"files directory.")
        
        if file_path.suffix.lower() not in MARKDOWN_EXTENSIONS:
            return f"Error: File '{file_path}' is not a Markdown file."
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return f"Content of '{file_path}':\n\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def create_markdown_file(file_name: str, content: str) -> str:
    """Create a new Markdown file in the files directory.
    Args:
        file_name (str): The name of the Markdown file to create.
        content (str): The content to write to the file.
    Returns:
        str: Success or error message.
    """
    try:
        # Always work in the files directory
        files_dir = Path(FILES_DIR)
        files_dir.mkdir(exist_ok=True)  # Ensure directory exists
        file_path = files_dir / file_name
        
        # Ensure the file has a .md extension
        if file_path.suffix.lower() not in MARKDOWN_EXTENSIONS:
            file_path = file_path.with_suffix('.md')
        
        # Check if file already exists
        if file_path.exists():
            return (f"Error: File '{file_path.name}' already exists. "
                    f"Use edit_markdown_file to modify existing files.")
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        return f"Successfully created Markdown file: '{file_path.name}'"
    except Exception as e:
        return f"Error creating file: {str(e)}"


@tool
def edit_markdown_file(file_name: str, content: str,
                       mode: str = "replace") -> str:
    """Edit an existing Markdown file in the files directory.
    Args:
        file_name (str): The name of the Markdown file to edit.
        content (str): The content to add or replace.
        mode (str): Edit mode - 'replace' (default), 'append', or 'prepend'.
    Returns:
        str: Success or error message.
    """
    try:
        # Always work in the files directory
        files_dir = Path(FILES_DIR)
        file_path = files_dir / file_name
        
        # Ensure .md extension if not provided
        if file_path.suffix.lower() not in MARKDOWN_EXTENSIONS:
            file_path = file_path.with_suffix('.md')
        
        if not file_path.exists():
            return (f"Error: File '{file_path.name}' does not exist. "
                    f"Use create_markdown_file to create new files.")
        
        if file_path.suffix.lower() not in MARKDOWN_EXTENSIONS:
            return f"Error: File '{file_path.name}' is not a Markdown file."
        
        if mode == "replace":
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            return f"Successfully replaced content in '{file_path.name}'"
        
        elif mode == "append":
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write('\n' + content)
            return f"Successfully appended content to '{file_path.name}'"
        
        elif mode == "prepend":
            with open(file_path, 'r', encoding='utf-8') as file:
                existing_content = file.read()
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content + '\n' + existing_content)
            return f"Successfully prepended content to '{file_path.name}'"
        
        else:
            return (f"Error: Invalid mode '{mode}'. "
                    f"Use 'replace', 'append', or 'prepend'.")
    
    except Exception as e:
        return f"Error editing file: {str(e)}"


@tool
def list_markdown_files() -> str:
    """List all Markdown files in the files directory.
    Returns:
        str: List of Markdown files found.
    """
    try:
        files_dir = Path(FILES_DIR)
        if not files_dir.exists():
            files_dir.mkdir(exist_ok=True)
            return f"No Markdown files found in '{FILES_DIR}' directory"
        
        if not files_dir.is_dir():
            return f"Error: '{FILES_DIR}' is not a directory."
        
        markdown_files = []
        for extension in MARKDOWN_EXTENSIONS:
            for file in files_dir.glob(f'*{extension}'):
                markdown_files.append(file.name)
        
        if not markdown_files:
            return f"No Markdown files found in '{FILES_DIR}' directory"
        
        return (f"Markdown files in '{FILES_DIR}' directory:\n" +
                "\n".join(f"- {file}" for file in sorted(markdown_files)))
    
    except Exception as e:
        return f"Error listing files: {str(e)}"


bedrock_model = BedrockModel(
    model_id="eu.anthropic.claude-sonnet-4-20250514-v1:0"
)

markdown_agent = Agent(
    description="An agent that can create, read, edit, and list Markdown files in the 'files' directory.",
    system_prompt="""You are a specialized Markdown assistant that helps
users manage Markdown files in the 'files' directory.

Your capabilities include:
1. Reading existing Markdown files from the files directory
2. Creating new Markdown files in the files directory
3. Editing existing Markdown files (replace, append, or prepend content)
4. Listing all Markdown files in the files directory
5. Using AI to help structure and improve Markdown content

Important: All file operations work within the 'files' directory.
Users only need to provide the filename (e.g., 'readme.md'), not the full path.

When users ask you to create or edit Markdown content, you should:
- Use proper Markdown syntax (headers, lists, links, code blocks, etc.)
- Structure content logically with appropriate headings
- Suggest improvements to make the content more readable
- Follow Markdown best practices

Always use the provided tools to perform file operations. When creating
content, make it well-structured and professional.""",
    tools=[read_markdown_file, create_markdown_file, edit_markdown_file,
           list_markdown_files],
    model=bedrock_model,
)


a2a_server = A2AServer(
    agent=markdown_agent,
    http_url="http://localhost:5002"
)
a2a_server.serve(host="0.0.0.0", port=5002)