import re
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Smart Product Matching", page_icon="🛍️", layout="wide")

# UI styling only: matching and data-processing logic remain unchanged.
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f5f8ff 0%, #ffffff 48%, #f7faff 100%);
        color: #172554;
    }
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: #172554;
    }
    .hero {
        padding: 1.7rem 1.8rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #102a56 0%, #1d4ed8 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 12px 30px rgba(29, 78, 216, 0.15);
    }
    .hero h1 {
        color: white !important;
        font-size: 2.1rem;
        margin: 0 0 .4rem 0;
    }
    .hero p {
        color: #e0eaff !important;
        font-size: 1rem;
        margin: 0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbeafe;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .045);
    }
    div[data-testid="stMetricLabel"] p {
        color: #52627a !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #1d4ed8 !important;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stFileUploader"] section {
        border-radius: 12px;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 11px;
        border: 0;
        background: #1d4ed8;
        color: white;
        font-weight: 600;
        padding: .55rem 1rem;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: #1e40af;
        color: white;
        border: 0;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #dbeafe;
        border-radius: 14px;
        overflow: hidden;
    }
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    .section-label {
        color: #64748b;
        font-size: .82rem;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-weight: 700;
        margin-bottom: .25rem;
    }
    .hint {
        color: #64748b;
        font-size: .92rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🤖 AI Smart Product Matching</h1>
    <p>Explore products, find closely related items, and compare listed prices.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_products():
    response = requests.get("https://dummyjson.com/products?limit=100", timeout=15)
    response.raise_for_status()
    products = response.json()["products"]
    df = pd.DataFrame(products).rename(columns={
        "title": "Product Name",
        "brand": "Brand",
        "price": "Price",
        "category": "Category",
        "thumbnail": "Image"
    })
    for column in ["Brand", "Category", "Image"]:
        if column not in df.columns:
            df[column] = ""
    df["Brand"] = df["Brand"].fillna("Unknown")
    df["Category"] = df["Category"].fillna("Other")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    # This public demo catalog does not provide individual seller names.
    df["Store"] = "Demo Catalog"
    return df.dropna(subset=["Product Name", "Price"])

try:
    all_products = load_products()
except Exception:
    st.error("Product catalogue could not be loaded. Check your internet connection and try again.")
    st.stop()

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"(\\d+)\\s*gb", r"\\1gb", text)
    text = re.sub(r"[^a-z0-9]", " ", text)
    return " ".join(text.split())

def model_id(text):
    text = str(text).lower().strip()
    samsung = re.search(r"\\b([mafs]\\d{2})\\b", text)
    if samsung:
        return samsung.group(1)
    iphone = re.search(r"\\biphone\\s*(\\d+)\\b", text)
    if iphone:
        return "iphone" + iphone.group(1)
    return None

def storage_id(text):
    match = re.search(r"(\\d+)\\s*gb", str(text).lower())
    return match.group(1) if match else None

# Exact product types prevent broad fuzzy matching such as Rice -> Wheat.
PRODUCT_TYPES = [
    ("Rice", ["rice", "basmati"]),
    ("Wheat/Atta", ["wheat", "atta", "flour"]),
    ("Salt", ["salt"]),
    ("Milk", ["milk"]),
    ("Face Wash/Cleanser", ["face wash", "cleanser"]),
    ("Moisturiser/Cream", ["moisturiser", "moisturizer", "moisturising cream", "moisturizing cream"]),
    ("Sunscreen", ["sunscreen", "spf"]),
    ("Shampoo", ["shampoo"]),
    ("Hair Oil", ["hair oil", "coconut oil"]),
    ("Phone Case/Cover", ["phone case", "phone cover", "case for"]),
    ("Charging Cable", ["charging cable", "usb cable", "cable"]),
    ("Charger", ["charger", "charging adapter"]),
    ("Earbuds/Headphones", ["earbuds", "earphones", "headphones", "headset"]),
    ("Smartwatch", ["smartwatch", "smart watch"]),
    ("Laptop", ["laptop", "notebook"]),
    ("Smartphone", ["smartphone", "mobile phone", "iphone", "galaxy"]),
]

def infer_product_type(text):
    value = normalize(text)
    # Prefer longer/more specific terms first, e.g. "face wash" before generic words.
    for product_type, keywords in PRODUCT_TYPES:
        if any(normalize(keyword) in value for keyword in keywords):
            return product_type
    return None

def row_product_type(text):
    return infer_product_type(text)

st.markdown('<div class="section-label">Explore and compare products</div>', unsafe_allow_html=True)
st.write("Browse the 100-product demo catalogue or search for a specific product to find related matches.")

categories = ["All Categories"] + sorted(all_products["Category"].dropna().unique().tolist())
selected_category = st.selectbox("Choose a product category", categories)

catalogue = all_products.copy()
if selected_category != "All Categories":
    catalogue = catalogue[catalogue["Category"] == selected_category].copy()

st.write(f"Products available: {len(catalogue)}")
query = st.text_input("Search product", placeholder="e.g. laptop, rice, face wash, phone case")

if query.strip():
    names = catalogue["Product Name"].fillna("").astype(str).tolist()
    if not names:
        st.warning("No products are available in this category.")
        st.stop()

    # Avoid vectorizer errors if all names/query contain no useful word features.
    texts = names + [normalize(query)]
    word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), stop_words="english")
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    word_matrix = word_vectorizer.fit_transform(texts)
    char_matrix = char_vectorizer.fit_transform(texts)
    word_scores = cosine_similarity(word_matrix[-1], word_matrix[:-1]).flatten()
    char_scores = cosine_similarity(char_matrix[-1], char_matrix[:-1]).flatten()
    scores = (0.6 * word_scores) + (0.4 * char_scores)

    results = catalogue.copy()
    results["Similarity (%)"] = (scores * 100).round(1)
    query_type = infer_product_type(query)
    query_model = model_id(query)
    query_storage = storage_id(query)

    # If the query identifies a product type, keep only products of that type.
    if query_type:
        results = results[
            results["Product Name"].astype(str).apply(row_product_type) == query_type
        ].copy()

    if query_model:
        results = results[
            results["Product Name"].astype(str).apply(
                lambda name: not model_id(name) or model_id(name) == query_model
            )
        ].copy()
    if query_storage:
        results = results[
            results["Product Name"].astype(str).apply(
                lambda name: not storage_id(name) or storage_id(name) == query_storage
            )
        ].copy()

    # Prefer literal matches when present; otherwise use similarity threshold.
    normalized_query = normalize(query)
    exact_matches = results[
        results["Product Name"].astype(str).apply(
            lambda name: normalized_query in normalize(name)
        )
    ].copy()
    if not exact_matches.empty:
        results = exact_matches
    elif not query_type:
        results = results[results["Similarity (%)"] >= 25].copy()

    results = results.sort_values(["Similarity (%)", "Price"], ascending=[False, True])
    st.subheader("🔎 Matching results")
    if results.empty:
        st.warning("No matching products found in the 100-product demo catalogue. Try a different search or category.")
    else:
        lowest_price = results.sort_values("Price").iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Matching products", len(results))
        col2.metric("Lowest listed price", f"₹{lowest_price['Price']:,.2f}")
        col3.metric("Listed source", str(lowest_price["Store"]))
        st.success(f"Lowest listed price: ₹{lowest_price['Price']:,.2f} at {lowest_price['Store']}")
        display_columns = [c for c in ["Product Name", "Brand", "Category", "Price", "Store", "Similarity (%)"] if c in results.columns]
        st.dataframe(results[display_columns], use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download comparison CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="smart_product_matches.csv",
            mime="text/csv",
        )
        st.caption("Prices are demo-catalogue prices, not live prices from multiple retailers.")

else:
    st.info("Enter a product name to find matching products.")
    st.subheader("🧾 Product catalogue")
    st.dataframe(catalogue, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("AI Smart Product Matching · TF-IDF and cosine similarity · 100-product demo catalogue")
