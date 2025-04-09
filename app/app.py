import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

@st.cache_resource
def load_ml_model():
    return load_model('model.h5')

model = load_ml_model()
class_names = ['biotite', 'bornite', 'chrysocolla', 'malachite', 
               'muscovite', 'pyrite', 'quartz']

st.title('Mineral Classification Expert System')
st.markdown("""
**Classify minerals from images**  
Trained on 5,640 mineral samples with 94.4% accuracy  
Supported minerals: Biotite, Bornite, Chrysocolla, Malachite, Muscovite, Pyrite, Quartz
""")

uploaded_file = st.file_uploader("Upload mineral image (JPG format)", type="jpg")

if uploaded_file:
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Model preprocessing
    processed_img = cv2.resize(image, (224, 224))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  # Convert for display
    normalized_img = (processed_img.astype(np.float32) / 255.0).reshape(1, 224, 224, 3)
    
    # Prediction
    prediction = model.predict(normalized_img)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_img, use_column_width=True, caption="Uploaded Image")
    with col2:
        st.subheader("Analysis Results")
        st.metric("Predicted Mineral", class_names[class_idx])
        st.metric("Confidence Level", f"{confidence:.1%}")
        st.progress(float(confidence))
        
        if confidence < 0.7:
            st.warning("Low confidence prediction - consider verification")
        else:
            st.success("High confidence prediction")
