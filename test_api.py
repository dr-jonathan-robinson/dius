import pandas as pd
import requests
import json
import time
import sys


API_BASE_URL = "http://localhost:5000" # use port 8000 if the API is running locally, port 5000 if running in a docker container.


def test_ping():
    response = requests.get(f"{API_BASE_URL}/ping")
    print(f"Health check status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_single_prediction(sample_data):
    response = requests.post(
        f"{API_BASE_URL}/predict", 
        json=sample_data
    )
    print(f"Single prediction status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_batch_prediction(batch_data):
    response = requests.post(
        f"{API_BASE_URL}/batch_predict", 
        json={"data": batch_data}
    )
    print(f"Batch prediction status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_csv_prediction(csv_file_path):
    with open(csv_file_path, 'rb') as f:
        files = {'file': (csv_file_path, f, 'text/csv')}
        response = requests.post(
            f"{API_BASE_URL}/batch_predict_csv", 
            files=files
        )
    
    print(f"CSV prediction status: {response.status_code}")
    
    if response.status_code == 200:
        output_file = "api_predictions.csv"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"Predictions saved to {output_file}")
        
        predictions = pd.read_csv(output_file)
        print(f"First 5 predictions:\n{predictions.head()}")
        
        return True
    else:
        print(f"Error response: {response.text}")
        return False


def do_all_the_things():
    """Main function to run all tests"""
    print("Testing DiUS ML Model API...")
    
    max_retries = 5
    for i in range(max_retries):
        if test_ping():
            break
        print(f"Server not ready, retrying in 2 seconds... ({i+1}/{max_retries})")
        time.sleep(2)
    else:
        print("Server not available after multiple attempts. Make sure it's running.")
        sys.exit(1)


    test_df = pd.read_csv('test.csv')
    print(f"Loaded test data with {len(test_df)} rows and {test_df.shape[1]} columns")
    

    if 'Y' in test_df.columns:
        X_test = test_df.drop('Y', axis=1)
    else:
        X_test = test_df
        

    sample = X_test.iloc[0].to_dict()
    
    batch = [row.to_dict() for _, row in X_test.iloc[:5].iterrows()]
    
    print("\n--- Testing Single Prediction ---")
    test_single_prediction(sample)
    
    print("\n--- Testing Batch Prediction (JSON) ---")
    test_batch_prediction(batch)
    
    print("\n--- Testing Batch Prediction (CSV) ---")
    test_csv_prediction('test.csv')
    
    print("\nAll tests completed!")
        


if __name__ == "__main__":
    do_all_the_things()
