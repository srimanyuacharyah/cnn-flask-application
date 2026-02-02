"""
Flask Application for Image Classification using CNN
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from model import predict_image, CLASS_LABELS

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Home page with image upload form"""
    return render_template('index.html', classes=CLASS_LABELS)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', 
                             error='No file uploaded', 
                             classes=CLASS_LABELS)
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return render_template('index.html', 
                             error='No file selected', 
                             classes=CLASS_LABELS)
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return render_template('index.html', 
                             error='Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, WEBP)', 
                             classes=CLASS_LABELS)
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        # Add timestamp to avoid filename conflicts
        import time
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        results = predict_image(filepath)
        
        # Render results page
        return render_template('result.html',
                             filename=filename,
                             predicted_class=results['predicted_class'],
                             confidence=results['confidence'],
                             top_predictions=results['top_predictions'])
    
    except Exception as e:
        return render_template('index.html', 
                             error=f'Error processing image: {str(e)}', 
                             classes=CLASS_LABELS)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for image prediction (returns JSON)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        import time
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = predict_image(filepath)
        results['image_url'] = f'/static/uploads/{filename}'
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("CNN Image Classification Flask App")
    print("=" * 50)
    print("\nSupported classes:", ", ".join(CLASS_LABELS))
    print("\nStarting server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
