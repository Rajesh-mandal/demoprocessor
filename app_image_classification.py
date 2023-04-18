import streamlit as st
#import tensorflow as tf
from keras.models import load_model
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import efficientnet.keras as efn

#Allow cache in the application for prediction 
@st.cache(allow_output_mutation=True)

#Load deep learning model 
def load_model():
  model=load_model('processoreffB2.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

#Frontend texts
st.markdown("<h1 style='text-align: center;'> Processor Defect classification</h1>", unsafe_allow_html=True)

instructions = """
                Please upload Processor images here.
                The image you select or upload will be fed through the 
                Deep Neural Network in real-time and the output will be displayed to the screen.
                """
st.write(instructions)

#File uploader with multiple upload option
file = st.file_uploader("Upload the image to be classified \U0001F447", type=["jpg", "png","jpeg"], accept_multiple_files=True)

st.set_option('deprecation.showfileUploaderEncoding', False)

#Function to pre-process the uploaded image and return prediction
def upload_predict(upload_image, model):
    
    size = (224,224)    
    image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
    img = np.expand_dims(image, 0).astype(np.float32) / 255.0
    preds = np.squeeze(model.predict(img)[0]) 
    print("preds: ", preds)
    return preds


if __name__ == '__main__':
  if file is None:
      st.text("Please upload an image file")
  else:
      #Loop through the List, "files", to classify every image uploaded
      for img_upload in (file):
          st.markdown("<h1 style='text-align: center;'>Here is the image you selected</h1>", unsafe_allow_html=True)
          image = Image.open(img_upload)
          st.image(image, use_column_width=True, caption=img_upload.name)
          predictions = upload_predict(image, model)
          clss_index = np.argmax(predictions)
          st.title("Here are the results")
          st.write('Predictions:')
          predictions_op = pd.DataFrame([predictions], columns = ["Bend_Pin","Good","Missing_Pins","Short_Pins"])
          st.write(predictions_op)
          class_names = ["Bend_Pin","Good","Missing_Pins","Short_Pins"]
          st.write("This image most likely belongs to **{}** category with a **{:.2f}** percent confidence."
              .format(class_names[clss_index], 100 * np.max(predictions)))
          #write to command prompt 
          print(
          "This image most likely belongs to {} with a {:.2f} percent confidence."
          .format(class_names[clss_index], 100 * np.max(predictions))
          )
