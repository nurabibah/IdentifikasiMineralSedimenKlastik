
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import base64 # Import base64 for embedding image in HTML

# Load the trained YOLOv8 model
# Make sure the model file exists in the correct path relative to app.py
model_path = './best.pt'
if not os.path.exists(model_path):
    st.error(f"Error: Model file not found at {model_path}. Please ensure your model file (e.g., best.pt) is in the same directory as app.py or update the model_path accordingly.")
else:
    try:
        model = YOLO(model_path)
        st.title("Deteksi Objek dengan YOLOv8")

        uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the image file
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

            # Perform inference
            results = model(image)

            # Display the results (image with bounding boxes)
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                st.image(im, caption="Hasil Deteksi", use_column_width=True)

                # --- Add functionality to display the number of detected objects ---
                num_objects = len(r.boxes)
                st.write(f"Jumlah objek terdeteksi: {num_objects}")
                # --- End of added functionality ---

                # --- Optional: Add functionality to save the result image as HTML ---
                # Check if there are detections before attempting to save/download
                if num_objects > 0: # Use the calculated num_objects
                    try:
                        # Save the result image temporarily
                        result_image_path = "result_image.png"
                        im.save(result_image_path)

                        # Read the image file and encode it in base64
                        with open(result_image_path, "rb") as img_file:
                            encoded_string = base64.b64encode(img_file.read()).decode()

                        # Create simple HTML content with the embedded image
                        # Correctly escape the inner f-string using single quotes and double curly braces
                        html_content = f'''
                        <!DOCTYPE html>
                        <html>
                        <head>
                        <title>Object Detection Result</title>
                        </head>
                        <body>
                        <h1>Hasil Deteksi Objek</h1>
                        <img src="data:image/png;base64,{encoded_string}" alt="Detected Image">
                        </body>
                        </html>
                        '''

                        # Provide a download link for the HTML file
                        st.download_button(
                            label="Unduh Hasil Deteksi (HTML)",
                            data=html_content,
                            file_name="deteksi_objek_result.html",
                            mime="text/html"
                        )
                        # Clean up the temporary image file
                        os.remove(result_image_path)
                    except Exception as e:
                        st.warning(f"Could not generate HTML download: {e}")
                else:
                     st.info("No objects detected to generate HTML download.")
                # --- End of added functionality ---


            # Optional: Display prediction details
            # st.write("Detail Prediksi:")
            # for r in results:
            #     for box in r.boxes:
            #         st.write(f"Class: {model.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}, Box: {box.xyxy[0].tolist()}")
    except Exception as e:
        st.error(f"An error occurred during model loading or inference: {e}")

