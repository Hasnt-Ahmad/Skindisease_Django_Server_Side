from imageio import imread
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model


def predictDisease(image_path):
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'polls', 'best_skin_model_efi_92_with_skin_.h5')
    disease_model = load_model(model_path)
    print(BASE_DIR)
    
    # predict_im = cv2.imread(image_path)
    predict_im = image_path
    predict_im=cv2.resize(predict_im,(224, 224))
    predict_im = np.expand_dims(predict_im, axis=0)

    # Make predictions
    preds = disease_model.predict(predict_im)

    # Define the class labels
    class_labels = ['Atopic','Basal','Benign','Melanocytic','Melanoma','Nail Fungus','Psoriasis','Seborrheic','Warts',"Normal Skin"]

    # Find the index of the class with maximum score
    top_3_indices = preds[0].argsort()[-3:][::-1]
    top_3_predictions = preds[0][top_3_indices]
    

    # Print the probabilities as percentages and the corresponding class labels
    results = [{"label": class_labels[top_3_indices[i]], "probability": float(f"{top_3_predictions[i] * 100:.3f}")} for i in range(3)]
    
    
    return results

    #print(class_labels[pred[0]])
    