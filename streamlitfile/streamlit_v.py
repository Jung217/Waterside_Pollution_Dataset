import cv2
import streamlit as st
from ultralytics import YOLO

def app():
    st.header('Object Detection Web App')
    st.subheader('Powered by YOLOv8')
    st.write('Welcome!')
    model = YOLO('best.pt')
    object_names = list(model.names.values())

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload video", type=['mp4'])
        selected_objects = st.multiselect('Choose objects to detect', object_names, default=['pollute']) 
        min_confidence = st.slider('Confidence score', 0.0, 1.0)
        st.form_submit_button(label='Submit')
            
    if uploaded_file is not None: 
        input_path = uploaded_file.name
        file_binary = uploaded_file.read()
        with open(input_path, "wb") as temp_file:
            temp_file.write(file_binary)
        video_stream = cv2.VideoCapture(input_path)

        if not video_stream.isOpened():
            st.error('Error opening video file.')
            return

        width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        fps = int(video_stream.get(cv2.CAP_PROP_FPS)) 
        output_path = input_path.split('.')[0] + '_output.mp4' 
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out_video.isOpened():
            st.error('Error opening output video file.')
            return

        with st.spinner('Processing video...'): 
            frame_count = 0
            while True:
                ret, frame = video_stream.read()
                if not ret:
                    break
                result = model(frame)
                for detection in result[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name = model.names[cls]
                    label = f'{object_name} {score}'

                    if object_name in selected_objects and score > min_confidence:
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x0, y0 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                detections = result[0].verbose()
                cv2.putText(frame, detections, (10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out_video.write(frame)
                frame_count += 1

            video_stream.release()
            out_video.release()
            st.success(f'Processing complete. Processed {frame_count} frames.')
            st.video(output_path)

if __name__ == "__main__":
    app()
