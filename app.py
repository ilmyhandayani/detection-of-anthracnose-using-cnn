#Import Library 
from flask import Flask, render_template, jsonify, request, g
import os, shutil, random
from shutil import copyfile

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, Nadam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model

#inisialisasi flask 
app = Flask(__name__)
app.secret_key = "cnn-classification"
model = ""

#mengatur rute pada tampilan web
@app.route('/', methods=['GET']) #get: http request
def index():
  data_normal = len(os.listdir('dataset/normal/'))
  data_antraknosa = len(os.listdir('dataset/antraknosa/'))
  data_lain = len(os.listdir('dataset/lainnya/'))
  jum_data = len(os.listdir('dataset/normal/')) + len(os.listdir('dataset/antraknosa/')) + len(os.listdir('dataset/lainnya/'))
  semua_data = [data_normal, data_antraknosa, data_lain, jum_data]
  return render_template('index.html', semua_data=semua_data)

@app.route('/training', methods=['GET']) #get: http request
def home():
  data_normal = len(os.listdir('dataset/normal/'))
  data_antraknosa = len(os.listdir('dataset/antraknosa/'))
  data_lain = len(os.listdir('dataset/lainnya/'))
  jum_data = len(os.listdir('dataset/normal/')) + len(os.listdir('dataset/antraknosa/')) + len(os.listdir('dataset/lainnya/'))
  semua_data = [data_normal, data_antraknosa, data_lain, jum_data]
  return render_template('training.html', semua_data=semua_data, display_training='none')

@app.route('/testing', methods=['GET']) #get: http request
def testing():
  data_normal = len(os.listdir('dataset/normal/'))
  data_antraknosa = len(os.listdir('dataset/antraknosa/'))
  data_lain = len(os.listdir('dataset/lainnya/'))
  jum_data = len(os.listdir('dataset/normal/')) + len(os.listdir('dataset/antraknosa/')) + len(os.listdir('dataset/lainnya/'))
  semua_data = [data_normal, data_antraknosa, data_lain, jum_data]
  return render_template('testing.html', semua_data=semua_data, display_testing='none')

@app.route('/upload', methods=['GET', 'POST']) #get & post: http request
def upload():
  if request.method == 'POST':
      f = request.files['file']
      f.save("static/testing/"+f.filename)
      hasil = main_program("static/testing/"+f.filename)
      data_normal = len(os.listdir('dataset/normal/'))
      data_antraknosa = len(os.listdir('dataset/antraknosa/'))
      data_lain = len(os.listdir('dataset/lainnya/'))
      jum_data = len(os.listdir('dataset/normal/')) + len(os.listdir('dataset/antraknosa/')) + len(os.listdir('dataset/lainnya/'))
      semua_data = [data_normal, data_antraknosa, data_lain, jum_data]

      return render_template('testing.html', hasil=hasil, gambar="static/testing/"+f.filename, semua_data=semua_data, display_testing='block')
    
@app.route('/tes', methods=['POST']) #post: http request 
def test():
    if request.method == "POST":
        #get data input 
        opti = get_value(request.form.getlist('optimizer'))
        loss = get_value(request.form.getlist('loss'))
        epoch = get_value(request.form.getlist('epochs'))
        print(epoch)
        epoch = int(epoch)

        #read data
        print("Jumlah Data Train Tiap Kelas")
        print('Jumlah gambar normal       :', len(os.listdir('dataset/normal/')))
        print('Jumlah gambar antraknosa   :', len(os.listdir('dataset/antraknosa/')))
        print('Jumlah gambar lainnya      :', len(os.listdir('dataset/lainnya/')))
        
        data_normal = len(os.listdir('dataset/normal/'))
        data_antraknosa = len(os.listdir('dataset/antraknosa/'))
        data_lain = len(os.listdir('dataset/lainnya/'))
        jum_data = len(os.listdir('dataset/normal/')) + len(os.listdir('dataset/antraknosa/')) + len(os.listdir('dataset/lainnya/'))
        semua_data = [data_normal, data_antraknosa, data_lain, jum_data]

        #cek direktori
        try:
          os.mkdir('/s/') 
        except:
          shutil.rmtree('/s/')
        
        #buat direktori
        os.mkdir('/s/')
        os.mkdir('/s/train/')
        os.mkdir('/s/val/')
        os.mkdir('/s/train/normal/')
        os.mkdir('/s/val/normal/')
        os.mkdir('/s/train/antraknosa/')
        os.mkdir('/s/val/antraknosa/')
        os.mkdir('/s/train/lainnya/')
        os.mkdir('/s/val/lainnya/')
        print("Berhasil buat direktori")
        
        #ratio split data
        train_ratio = 0.8

        #pembagian Training dan Validasi untuk dataset normal
        source_00 = 'dataset/normal/'
        train_00 = '/s/train/normal/'
        val_00 = '/s/val/normal/'
        train_val_split(source_00, train_00, val_00, train_ratio)

        #pembagian Training dan Validation untuk dataset antraknosa
        source_01 = 'dataset/antraknosa/'
        train_01 = '/s/train/antraknosa/'
        val_01 = '/s/val/antraknosa/'
        train_val_split(source_01, train_01, val_01, train_ratio)

        #pembagian Training dan Validation untuk lainnya
        source_02 = 'dataset/lainnya/'
        train_02 = '/s/train/lainnya/'
        val_02 = '/s/val/lainnya/'
        train_val_split(source_02, train_02, val_02, train_ratio)
        
        #menampung data list dari split data
        data = [[len(os.listdir(source_00)),len(os.listdir(train_00)),len(os.listdir(val_00))],
        [len(os.listdir(source_01)),len(os.listdir(train_01)),len(os.listdir(val_01))],
        [len(os.listdir(source_02)),len(os.listdir(train_02)),len(os.listdir(val_02))]]

        #plotting
        label = ['All', 'Training', 'Validation']
        X = np.arange(3)
        print(X+0.25)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(X+0.00,data[0],color='r',width=0.25)
        ax.bar(X+0.25,data[1],color='g',width=0.25)
        ax.bar(X+0.50,data[2],color='b',width=0.25)

        plt.xticks(X+0.25, label)
        plt.title('Split Data')
        plt.legend(['Normal','Antraknosa','Lainnya'],loc='best')
        plt.savefig('static/images/SplitData.png', bbox_inches='tight')
        plt.close()
        
        #data generator
        train_datagen = ImageDataGenerator(
                  rescale = 1./255.,
                  rotation_range = 30,
                  horizontal_flip = True,
                  shear_range = 0.3,
                  fill_mode = 'nearest',
                  width_shift_range = 0.2,
                  height_shift_range = 0.2,
                  zoom_range = 0.1
        )

        val_datagen = ImageDataGenerator(
                        rescale = 1./255.,
                        rotation_range = 30,
                        horizontal_flip = True,
                        shear_range = 0.3,
                        fill_mode = 'nearest',
                        width_shift_range = 0.2,
                        height_shift_range = 0.2,
                        zoom_range = 0.1
        )

        train_dir = '/tmp/s/train/'
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size = (150, 150),
            batch_size = 40, 
            class_mode = 'categorical'
        )

        val_dir = '/tmp/s/val/'
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size = (150, 150),
            batch_size = 10,
            class_mode = 'categorical'
        )

        # callbacks
        class myCallback(tf.keras.callbacks.Callback):
          def on_epoch_end(self, epoch, logs = {}):
            if(logs.get('accuracy') > 0.99):
              print('\nAkurasi mencapai 99%')
              self.model.stop_training = True

        callbacks = myCallback()
        
        #membuat model
        model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
          tf.keras.layers.MaxPooling2D(2, 2),
          tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
          tf.keras.layers.MaxPooling2D(2, 2),
          tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
          tf.keras.layers.MaxPooling2D(2, 2),
          tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
          tf.keras.layers.MaxPooling2D(2, 2),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(200, activation = 'relu'),
          tf.keras.layers.Dropout(0.3,seed=112),
          tf.keras.layers.Dense(500, activation = 'relu'),
          tf.keras.layers.Dropout(0.5,seed=112),
          tf.keras.layers.Dense(3, activation = 'softmax')
        ])
        
        model.summary()

        #compile model 
        model.compile(loss = loss, 
              optimizer = opti,
              metrics = ['accuracy'])
        
        history = model.fit(
            train_generator,
            steps_per_epoch = 20,
            epochs = epoch, 
            validation_data = val_generator,
            validation_steps = 5,
            verbose = 1, 
            callbacks = [callbacks]
        )

        epc = []
        for i in range(len(history.history['loss'])):
          epc.append(f"Epoch {i+1}/15\n29/29 [==============================] - loss: {history.history['loss'][i]:.4f} - accuracy: {history.history['accuracy'][i]:.4f} - val_loss: {history.history['val_loss'][i]:.4f} - val_accuracy: {history.history['val_accuracy'][i]:.4f}")

        evaluate = model.evaluate(val_generator)
        ac = evaluate[1]
        los = evaluate[0]
        print('accuracy : ',ac)
        print('loss : ', los)

        #menghitung akurasi dan menampilkan akurasi
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        # print('accuracy: ', acc)
        # print('val_accuracy: ', val_acc)
        # print('loss: ', loss)
        # print('val_loss: ', val_loss)

        epochs = range(len(acc))

        #membuat plotting untuk hasil Akurasi 
        plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
        plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
        plt.title('Training and Validation accuracy')
        plt.legend(loc = 'best')
        plt.savefig('static/images/Accuracy.png')
        plt.close()
        
        #membuat plotting untuk hasil Loss 
        plt.plot(epochs, loss, 'r', label = 'Training Loss')
        plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend(loc = 'best')
        plt.savefig('static/images/Loss.png')
        plt.close()

        #save model 
        save_dir=r'/static'
        save_id=str ('model_2.h5')
        save_loc=os.path.join(save_dir, save_id)
        model.save(save_loc)
        model.save('model_2.h5')

        return render_template('training.html', semua_data=semua_data,ac=ac,los=los,epoch=epc, display_training='block')

def get_value(data):
  for i in data:
    if i != '':
      return i

def train_val_split(source, train, val, train_ratio):
  total_size = len(os.listdir(source))
  train_size = int(train_ratio * total_size)
  val_size = total_size - train_size

  randomized = random.sample(os.listdir(source), total_size)
  train_files = randomized[0:train_size]
  val_files = randomized[train_size:total_size]

  for i in train_files:
    i_file = source + i
    destination = train + i
    copyfile(i_file, destination)

  for i in val_files:
    i_file = source + i
    destination = val + i
    copyfile(i_file, destination)

def main_program(fn):
  # predicting images
  modelcnn = load_model('model_2.h5')
  path = fn
  img = image.load_img(path, target_size = (150, 150))
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis = 0) #axis 0 is row 

  images = np.vstack([x])
  classes = modelcnn.predict(images, batch_size = 50)

  print(fn)

  class_list = ['lainnya','antraknosa','normal']
  # class_list = os.listdir('/tmp/dataset/')
  print(class_list)
    
  for j in range(3):
    if classes[0][j] == 1. :
      return f'This image belongs to class {class_list[j]}'
      break


app.run(host='127.0.0.1', port='1212', debug=True)