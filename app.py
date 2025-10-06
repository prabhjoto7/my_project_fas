import streamlit as st
import pickle
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ------------------------------
# Load precomputed embeddings
# ------------------------------
with open('fashion_embeddings2.pkl', 'rb') as f:
    data = pickle.load(f)

image_embeddings = data['embeddings']
image_paths = data['image_paths']         # full paths
image_categories = data['categories']

with open("fashion_embeddings_text.pkl", "rb") as f:
    text_data = pickle.load(f)

text_embeddings = text_data["text_embeddings"]
titles = text_data["texts"]
text_image_paths = text_data["image_paths"]   # full paths
text_categories = text_data["categories"]

# ------------------------------
# Load models
# ------------------------------
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Fashion Hybrid Recommendation System")

search_method = st.radio("Choose search method:", ["By Image", "By Search"])

uploaded_file = None
search_query = None

if search_method == "By Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
elif search_method == "By Search":
    search_query = st.text_input("Enter your search query")

# ------------------------------
# Helper functions
# ------------------------------
def get_image_embedding(img_input, model=resnet_model, target_size=(224,224)):
    """Get embedding for uploaded image or path"""
    try:
        if isinstance(img_input, str):
            img = image.load_img(img_input, target_size=target_size)
        else:
            img = Image.open(img_input).resize(target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        emb = model.predict(x, verbose=0)
        return emb.flatten()
    except Exception as e:
        st.error(f"Error generating image embedding: {e}")
        return None

def recommend_by_image(uploaded_img, embeddings, paths, categories, top_k=5):
    query_emb = get_image_embedding(uploaded_img)
    if query_emb is None:
        return []
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [(paths[i], categories[i], sims[i]) for i in top_idx]

def recommend_by_text(query, text_embeddings, titles, paths, categories, top_k=5):
    query_emb = text_model.encode([query])[0]
    sims = cosine_similarity([query_emb], text_embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [(paths[i], categories[i], titles[i], sims[i]) for i in top_idx]

def load_image_safe(img_path):
    """Load image safely, return placeholder if not found"""
    img_path = Path(img_path)
    if not img_path.exists():
        img_path = Path.cwd() / img_path
    try:
        return Image.open(img_path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224,224), color="gray")

# ------------------------------
# Display results
# ------------------------------
if search_method == "By Image" and uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=250)
    results = recommend_by_image(uploaded_file, image_embeddings, image_paths, image_categories, top_k=5)

    if results:
        st.subheader("Top Recommendations")
        cols = st.columns(len(results))
        for i, (img_path, cat, sim) in enumerate(results):
            img = load_image_safe(img_path)
            cols[i].image(img, caption=f"{cat}\nSimilarity: {sim:.2f}", use_container_width=True)
    else:
        st.warning("No recommendations found.")

elif search_method == "By Search" and search_query:
    results = recommend_by_text(search_query, text_embeddings, titles, text_image_paths, text_categories, top_k=5)

    if results:
        st.subheader(f"Results for '{search_query}'")
        cols = st.columns(len(results))
        for i, (img_path, cat, title, sim) in enumerate(results):
            img = load_image_safe(img_path)
            cols[i].image(img, caption=f"{title}\n{cat}\nSimilarity: {sim:.2f}", use_container_width=True)
    else:
        st.warning("No matching products found.")
