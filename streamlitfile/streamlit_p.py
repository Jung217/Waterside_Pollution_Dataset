import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

def app():
    st.header('Object Detection Web App')
    st.subheader('Powered by YOLOv9')
    st.write('This is demonstration about pollution detection model.')
    model = YOLO("best.pt")
    object_names = list(model.names.values())

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
        selected_objects = st.multiselect('Choose objects to detect', object_names, default=['pollute']) 
        min_confidence = st.slider('Confidence score', 0.0, 1.0)
        st.form_submit_button(label='Submit')
        
    if uploaded_file is not None:
        # Read and process the uploaded image
        image = uploaded_file.read()
        nparr = np.frombuffer(image, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform detection
        result = model(frame)
        
        # Draw results on the image
        for detection in result[0].boxes.data:
            x0, y0 = (int(detection[0]), int(detection[1]))
            x1, y1 = (int(detection[2]), int(detection[3]))
            score = round(float(detection[4]), 2)
            cls = int(detection[5])
            object_name = model.names[cls]
            label = f'{object_name} {score}'

            if model.names[cls] in selected_objects and score > min_confidence:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.putText(frame, label, (x0, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Convert image to RGB for display in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption='Processed Image', use_column_width=True)

if __name__ == "__main__":
    app()
