import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pymongo
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename

app = Flask(__name__)
# app.secret_key = 'your_secret_key'

# MongoDB connection
url = 'mongodb+srv://nikhilarora13832:nikhil123@cluster0.qwayyb4.mongodb.net/'
client = pymongo.MongoClient(url)
db = client['test_mongo']
users_collection = db['users']
data_collection = db['data']

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    if 'username' in session:
        return render_template('home.html')
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    text = request.form['text']
    image = request.files['image']
    
    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        
        # Store the data in MongoDB
        data_collection.insert_one({
            "username": session['username'],
            "text": text,
            "image_path": image_path
        })
        
        flash('Data uploaded successfully!', 'success')
    else:
        flash('Image upload failed', 'danger')
    
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users_collection.find_one({"username": username})
        
        if user and user['password'] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if users_collection.find_one({"username": username}):
            flash('Username already exists', 'danger')
        else:
            users_collection.insert_one({"username": username, "password": password})
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
