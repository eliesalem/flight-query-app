from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import sys
import experiment


app = Flask(__name__)
CORS(app)

# Full path to the Python script
script_path = '/Users/eliesalem/Desktop/flight_query_folder/experiment.py'

# This function ensures that Python runs in the correct environment
def run_python_script_in_env(script_path, query):
    print("running python script in environ")
    # Specify the full path to the Python interpreter in your virtual environment
    python_executable = "/opt/homebrew/Caskroom/miniconda/base/envs/cs187/bin/python"
    
    # Run the Python script with the correct environment and pass the query as an argument
    result = subprocess.run([python_executable, script_path, query], capture_output=True, text=True)
    
        # Print both stdout and stderr for debugging
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    
    # Capture stdout and stderr
    if result.returncode == 0:
        print("returning standard out")
        return result.stdout
    else:
        print("returning standard error")
        return f"Error: {result.stderr}"

@app.route('/query', methods=['POST'])
def handle_query():
    query = request.json.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400  # Handle missing query case
    
    # Disassemble the query using your Python script
    output = run_python_script_in_env(script_path, query)

    #print("Return string: ", experiment.return_string)
    # Return the result as a JSON response
    return jsonify({'result': output})

if __name__ == '__main__':
    test_query = 'i would like a flight between boston and dallas'
    output = run_python_script_in_env(script_path, test_query)
    print(output)
    app.run(debug=True)