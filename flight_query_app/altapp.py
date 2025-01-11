from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import requests  # Importing the requests library

# Initialize the FastAPI app
app = FastAPI()
print("FastAPI app created:", app)

# Full path to the Python script
script_path = '/Users/eliesalem/Downloads/project-4-niels/experiment.py'

# This function ensures that Python runs in the correct environment
def run_python_script_in_env(script_path, query):
    # Specify the full path to the Python interpreter in your virtual environment
    python_executable = "/opt/homebrew/Caskroom/miniconda/base/envs/cs187/bin/python"
    
    # Run the Python script with the correct environment and pass the query as an argument
    print(f"Running script at {script_path} with query: {query}")  # Debugging line
    result = subprocess.run([python_executable, script_path, query], capture_output=True, text=True)
    
    # Print both stdout and stderr for debugging
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    
    # Capture stdout and stderr
    if result.returncode == 0:
        return result.stdout
    else:
        return f"Error: {result.stderr}"

# Define the request model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def handle_query(request: QueryRequest):
    query = request.query
    
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")  # Handle missing query case
    
    # Log the received query for debugging
    print(f"Received query: {query}")
    
    # Disassemble the query using your Python script
    output = run_python_script_in_env(script_path, query)
    
    # Return the result as a JSON response
    return {"result": output}

# Run a test query when starting the server
if __name__ == '__main__':
    test_query = 'i would like a flight between boston and dallas'
    output = run_python_script_in_env(script_path, test_query)
    print("Test query result:", output)

    # Python `requests` code to send a POST request to the FastAPI app
    print("Sending test request to FastAPI...")
    url = "http://127.0.0.1:8000/query"
    headers = {"Content-Type": "application/json"}
    data = {"query": "i would like a flight between boston and dallas"}
    
    # Send the POST request
    response = requests.post(url, json=data, headers=headers)
    
    # Print the response from FastAPI
    print(f"Response from FastAPI: {response.json()}")
