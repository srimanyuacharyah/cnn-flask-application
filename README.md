# CNN Flask Image Classification Application

A web application for image classification using a Convolutional Neural Network (CNN) built with TensorFlow/Keras and served via Flask.

## Features

- 🧠 **CNN Model**: Deep learning model trained on CIFAR-10 dataset
- 📁 **Drag & Drop Upload**: Modern file upload interface
- 📊 **Confidence Scores**: Displays prediction confidence and top-3 predictions
- 🎨 **Premium Dark UI**: Beautiful glassmorphism design with animations
- 🔌 **REST API**: JSON endpoint for programmatic access

## Supported Classes

The model can classify images into 10 categories:
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open in browser**: Navigate to `http://127.0.0.1:5000`

## First Run

On first run, the application will automatically:
1. Download the CIFAR-10 dataset
2. Train a CNN model (~10 epochs)
3. Save the trained model to `model/cnn_model.h5`

This initial training takes a few minutes. Subsequent runs will load the saved model.

## API Usage

### POST /api/predict

Upload an image for classification:

```bash
curl -X POST -F "file=@image.jpg" http://127.0.0.1:5000/api/predict
```

**Response**:
```json
{
  "predicted_class": "cat",
  "confidence": 92.5,
  "top_predictions": [
    {"class": "cat", "confidence": 92.5},
    {"class": "dog", "confidence": 5.2},
    {"class": "deer", "confidence": 1.3}
  ],
  "image_url": "/static/uploads/1234567890_image.jpg"
}
```

## Project Structure

```
cnn-flask-application/
├── app.py              # Flask application
├── model.py            # CNN model and prediction utilities
├── requirements.txt    # Python dependencies
├── model/              # Saved CNN model
├── static/
│   ├── css/style.css   # Styling
│   └── uploads/        # Uploaded images
└── templates/
    ├── index.html      # Upload page
    └── result.html     # Results page
```

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- Flask 2.0+
- Pillow 9.0+

## License

MIT License