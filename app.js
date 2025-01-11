async function submitQuery() {
    // Get the query input
    const query = document.getElementById('queryInput').value;
    
    // Check if the query is empty
    if (!query.trim()) {
        alert("Please enter a query!");
        return;
    }
    
    // Show loading message
    document.getElementById('loadingMessage').style.display = 'block';
    document.getElementById('response').style.display = 'none';
    
    try {
        // Send the query to the Flask backend
        const response = await fetch('http://localhost:5000/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        // Hide the loading message and display the result
        document.getElementById('loadingMessage').style.display = 'none';
        document.getElementById('response').style.display = 'block';

        if (data.error) {
            document.getElementById('responseText').textContent = `Error: ${data.error}`;
        } else {
            document.getElementById('responseText').textContent = data.result;
        }

    } catch (error) {
        console.error("Error:", error);
        document.getElementById('loadingMessage').style.display = 'none';
        document.getElementById('response').style.display = 'block';
        document.getElementById('responseText').textContent = "An error occurred while processing the request.";
    }
}
