import base64
import json
import os
import requests
import subprocess

def get_access_token():
    result = subprocess.run(['gcloud', 'auth', 'print-access-token'], 
                          capture_output=True, text=True)
    return result.stdout.strip()

def process_document(file_path):
    # Read the file and encode it
    with open(file_path, 'rb') as file:
        content = base64.b64encode(file.read()).decode('utf-8')
    
    # Prepare the request
    url = "https://us-documentai.googleapis.com/v1/projects/969504446715/locations/us/processors/ddc065df69bfa3b5:process"
    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json"
    }
    data = {
        "rawDocument": {
            "content": content,
            "mimeType": "application/pdf"
        }
    }
    
    # Make the request
    response = requests.post(url, headers=headers, json=data)
    return response.json()

if __name__ == "__main__":
    # Use absolute path with leading space in filename
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "test_documents", " 2025.01.31 - 10 a. Northzone X L.P. - Drawdown 19 - Icons Partnership.pdf")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit(1)
    
    print(f"Processing file: {file_path}")
    result = process_document(file_path)
    print(json.dumps(result, indent=2)) 