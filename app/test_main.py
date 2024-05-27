from main import app
from fastapi.testclient import TestClient
from colorama import Fore

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fast API is running"}

    print(f"{Fore.GREEN} Passed")
    
def test_get_query_answer_endpoint():
    query = "reduce the risk of developing diet-related chronic disease?"
    response = client.post("/", json={"query": query})
    assert response.status_code == 200
    assert response.json()["answer"] != ""
    
    print(f"{Fore.GREEN} Passed")
    
if __name__ == "__main__":
    test_root_endpoint()
    test_get_query_answer_endpoint()