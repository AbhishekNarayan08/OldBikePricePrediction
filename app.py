from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('used_bikes')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/prediction", methods = ["POST"])
def prediction():

    city = request.form['city']
    kms = request.form['kms_driven']
    kms_driven = float(kms)
    owner = request.form['owner']
    brand = request.form['brand']
    ag = request.form['age']
    age = float(ag)
    pow = request.form['power']
    power = float(pow)
    inp = { "city": city, "kms_driven": kms_driven, "owner": owner, "brand": brand, "age": age, "power": power}
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in inp.items()}
    # print(inp)
    # print(input_dict)
    pred = model.predict(input_dict)
    return render_template("prediction.html", data = pred[0][0])

if __name__ == "__main__":
    app.run(debug = True)
