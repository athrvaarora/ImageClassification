from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("model/your_model.h5")  # Update with your actual model file

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img = tf.image.decode_image(file.read(), channels=3)
    img = tf.image.resize(img, [224, 224])  # Resize as per model requirements
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    class_label = prediction.argmax()  # Replace with your classification logic

    return jsonify({"class": str(class_label)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
