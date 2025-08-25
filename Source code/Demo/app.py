from generate_similarities import search_similar_images
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import time

# /dataset/images includes cutout and model
DATASET_PATH = "./dataset/model-img/model"
# DATASET_PATH2 = "./dataset/cutout-img/cutout"

def request_images(image_path):
    response = requests.get(image_path)
    image = Image.open(BytesIO(response.content))
    return image

def load_similar_images(query_image):
    results = search_similar_images(query_image)
    for result in results:
        print(f"========== Model: {result['name']} ==========")
        st.title(f"Top 10 Searching Results With {result['name']}")
        rows = [st.columns(5) for _ in range(5)]
        cols = [col for row in rows for col in row]
        for col, path, brand, score in zip(cols, result["paths"], result["brands"], result["scores"]):
            # image = request_images(path)
            file_name = path.split("/")[-1]
            image = Image.open(DATASET_PATH + "/" + file_name)
            col.image(image, caption=f"Brand: {brand}", width=200)
            print(f"{file_name} - score: {score}")
        print(f"TOTAL COMPUTING TIME FOR {result['name']}: {result['time']}")

if __name__ == "__main__":

    st.set_page_config(
        page_title="Fashion Image Search Engine",
        layout="wide",
    )
    st.markdown(
        """
        <style>
            .larger-font {
                font-size: 24px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Fashion Image Search Engine")

    input_url = st.text_input(
        "Searching Similar Images By URL",
        placeholder="Enter your image URL here..."
    )
    button_clicked = st.button("Search")
    if button_clicked:
        try:
            # response = requests.get(input_url)
            # query_image = Image.open(BytesIO(response.content))
            query_image = request_images(input_url)
            st.image(query_image, width=300)
            with st.spinner("Searching Images Similar To Your Input..."):
                time.sleep(20)
            load_similar_images(query_image)
        except:
            st.write("Invalid URL! Please try again!")