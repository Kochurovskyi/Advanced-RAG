#!/usr/bin/env python3
"""
Script to convert graph.mmd to graph.png
"""
import requests
import base64

def convert_mermaid_to_png():
    """Convert graph.mmd to graph.png using Mermaid Live API"""
    try:
        # Read the Mermaid content
        with open("graph.mmd", "r", encoding="utf-8") as f:
            mermaid_content = f.read()
        
        print("Converting graph.mmd to PNG...")
        
        # Use Mermaid Live API
        url = "https://mermaid.ink/img/"
        
        # Encode the Mermaid content
        encoded_content = base64.urlsafe_b64encode(mermaid_content.encode('utf-8')).decode('ascii')
        
        # Make request to get PNG
        response = requests.get(f"{url}{encoded_content}")
        
        if response.status_code == 200:
            # Save PNG file
            with open("graph.png", "wb") as f:
                f.write(response.content)
            
            print("Successfully generated graph.png")
            return True
        else:
            print(f"Error: HTTP {response.status_code}")
            return False
        
    except Exception as e:
        print(f"Error converting Mermaid to PNG: {e}")
        return False

if __name__ == "__main__":
    convert_mermaid_to_png()