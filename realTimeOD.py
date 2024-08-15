from ultralytics import YOLO
import cv2

model = YOLO("YOUR.pt") # Modify HERE

#results = model(source="IMG_4162.MOV",show = True ,conf = 0.4, save=True)

# Capture video from webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(source=frame, conf=0.4, save=False)

    # Display the results
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO-World Real-Time Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()