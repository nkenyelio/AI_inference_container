import yaml
from minio import Minio
from datetime import datetime

from pypyr import pipelinerunner


from sklearn.metrics import classification_report
from werkzeug.utils import secure_filename, redirect

from dataloader import DataLoader_inference

from abc import ABC, abstractmethod
from utils.config import Config

# initialize our Flask application and the Keras model

UPLOAD_FOLDER = "data"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'log'}
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = None

df = []


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    global df
    # model = ResNet50(weights="imagenet")
    # model = load_model("lstm.h5")
    model = tf.keras.models.load_model('lstm.h5')


with open ("config_aws_model.yaml", "r") as stream:
    config = yaml.safe_load(stream)




# You can run a pipeline via the API like this:
context = pipelinerunner.run(pipeline_name='pipeline-dir/my-pipe',
                             args_in=['arbkey=pipe', 'anotherkey=song'])


# context behaves like a dict
print(context['myoutput'])  # I was set in the pipeline!
print(context['input_values'][0])  # pipe
print(context['some_nesting']['down_level']['arb_number'])  # 123


def download_model_aws(filename):
    client = Minio(
        "minio.mlops.svc.cluster.local:9000",
        access_key=config["MINIO_USER"],
        secret_key=config["MINIO_PASSWORD"],
        secure=False
    )
    found = client.bucket_exists("mhealth")
    if not found:
        client.make_bucket("mhealth")
    else:
        print("Bucket for model 'mhealth' already exists")

    print("File successfully uploaded!")  
    # You can run a pipeline via the API like this:
    context = pipelinerunner.run(pipeline_name='config')


    # context behaves like a dict
    print(context['myoutput'])  # I was set in the pipeline!
    print(context['input_values'][0])  # pipe
    print(context['some_nesting']['down_level']['arb_number'])  # 123


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("file"):
            # read the file provided
            file = flask.request.files["file"]
            if not file:
                return "No file"

            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flask.flash('No selected file')
                return redirect(flask.request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Write into file
            #subject = 1
            #filename = secure_filename(file.filename)
            #os.rename(filename, 'mHealth_subject' + str(subject) + '.log')
            #filename = 'mHealth_subject' + str(subject) + '.log'
            # file_name = 'mHealth_subject' + str(subject) + '.log'
            #os.path.join('data', filename)

            # DataLoader.read_subject(1)

            # preprocess the image and prepare it for classification
            X_test, Y_label = prepare_datapoint(1, 100)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            y_pred = model.predict(X_test)

            # 12 total classes

            labs = set()
            labs.add(0)
            labs.add(1)
            labs.add(2)
            labs.add(3)
            labs.add(4)
            labs.add(5)
            labs.add(6)
            labs.add(7)
            labs.add(8)
            labs.add(9)
            labs.add(10)
            labs.add(11)
            labs.add(12)
            preds = []
            new_test = []

            # converting one hot prediction  and real label to single interger value

            for i, p in enumerate(y_pred):
                preds.append(np.argmax(p))
                new_test.append(np.argmax(Y_label[i]))

            y_pred = preds
            Y_label = new_test

            # classification reports

            classes = ["Null class", "Standing still", "Sitting and relaxing", "Lying down", "Walking",
                       "Climbing stairs", "Waist bends forward", "Frontal elevation of arms",
                       "knees bending (crouching)", "Cycling", "Running", "Jump front & back"]

            report_class = classification_report(Y_label, y_pred, target_names=classes, output_dict=True)

            #results = json.dumps(report_class, indent=4)


            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions

            data["predictions"].append(report_class)
            # indicate that the request was a success

            data["success"] = True

    # return the data dictionary as a JSON response
    #return flask.jsonify(data)
    return flask.json.dumps(data, indent=4)


if __name__ == '__main__':
    filename = f"data-{datetime.now().date()}.config_aws_model"
    try:
        download_model_aws(filename)
    except Exception as e:
        print(e)
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()