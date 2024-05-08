### Building Url Dynamically 
####Variable Rules And URL Building

from flask import Flask,request
app=Flask(__name__)

@app.route('/' , methods=["GET"])
def welcome():
    return 'Hello World'

# Route with parameters
@app.route('/user/<username>')
def user_profile(username):
    return f'User Profile: {username}'


# Route with query string
@app.route('/search')
def search():
    keyword = request.args.get('name')
    return f'Searching for: {keyword}'

# Route with data in body 
@app.route('/json', methods=['POST'])
def json_endpoint():
    data = request.json
    return f'Received JSON data: {data}'

@app.route('/form', methods=['POST'])
def form_endpoint():
    data = request.form
    return f'Received Form data: {data}'

@app.route('/raw', methods=['POST'])
def raw_data_endpoint():
    data = request.data
    return f'Received raw data: {data}'


# Middleware 
@app.before_request
def before_auth_request_func():
    if request.endpoint == 'auth' and request.method == 'POST':
        # Perform any checks or operations specific to the /auth route
        user = request.form.get('name')
        if not user:
            return 'Unauthorized', 401  # Return unauthorized response if user is not provided

# Route for authentication
@app.route('/auth', methods=["POST"])
def auth():
    user = request.form.get('name')
    return f'Welcome {user}'





if __name__=='__main__':
    app.run(debug=True)