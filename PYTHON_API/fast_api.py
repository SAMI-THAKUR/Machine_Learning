from fastapi import FastAPI,responses,HTTPException
from pydantic import BaseModel

# Create an instance of the FastAPI class
app = FastAPI()

# Define a route using a decorator
@app.get("/")
def read_root():
    # return 'sami'
    return responses.PlainTextResponse('SAMI THAKUR')

# Run the app using uvicorn with auto-reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    # cmd: uvicorn file_name:app --reload



# In-memory database for demonstration
items = []

@app.get('/items/')
async def get_items():
    return {"items": items}


# Endpoint to create a new item with POST request
@app.post("/items/")
async def create_item(data: str):
    if data in items:
        raise HTTPException(status_code=400, detail="Item already exists")
    items.append(data)
    return {"message": "Item created successfully", "item": data}

# Endpoint to update an existing item with PUT request
@app.put("/items/{name}")
async def update_item(name: str):
    if name not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    items[items.index(name)] = name
    return {"message": "Item updated successfully", "name":name}