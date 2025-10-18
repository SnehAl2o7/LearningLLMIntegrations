import pandas as pd
import numpy as np


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # Open-source alternative
import gradio as gr


books = pd.read_csv("books_with_emotion.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "images/Gemini_Generated_Image_t5adllt5adllt5ad.png",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, HuggingFaceEmbeddings())

def retrieve_semantic(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k : int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    
    else:
        book_recs = book_recs.head(final_top_k)

    
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    
    elif tone == "Suprise":
        book_recs.sort_values(by="suprise", ascending=False, inplace=True)

    elif tone == "Anger":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)

    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)

    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    

    return book_recs.head(final_top_k)



def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendation = retrieve_semantic(query, category, tone)
    results = []

    for _, row in recommendation.iterrows():
        description = row["description"]
        truncated_desc = description.split()
        truncated_description = " ".join(truncated_desc[:30]) + "...."

        authors_split = row["authors"].split(";")

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        
        caption = f"{row['title']} by {authors_str} : {truncated_description}"

        results.append((row["large_thumbnail"], caption))
    
    return results



categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Suprise", "Anger", "Suspenseful" , "Sad"]

#the changing of theme

custom_theme = gr.themes.Glass().set(
    primary_700="#7c0786", 
    border_radius_xl="10px", 
)

with gr.Blocks(theme = custom_theme) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        gr.Column(scale=1) 
        
        with gr.Column(scale=4):
            gr.Markdown("## 📖 Enter Your Query and Preferences")
            
            user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder= "e.g., A story about a detective in 1920s London.")
            
            with gr.Row():
                tone_dropdown = gr.Dropdown(choices=tones, label="Select an Emotional Tone:", value="All")
                category_dropdown = gr.Dropdown(choices=categories, label= "Select a Category:", value="All")
            
            submit_button = gr.Button("🔍 Find Recommendations", variant="primary")
            
        gr.Column(scale=1)
        
    gr.Markdown("---") 
    gr.Markdown("## ✨ Personalized Book Recommendations")
    
    output = gr.Gallery(label="Recommmending books", columns=8, rows=2, object_fit="contain")

    submit_button.click(fn=recommend_books,
                        inputs = [user_query, tone_dropdown, category_dropdown],
                        outputs= output)
    

if __name__ == "__main__":
    dashboard.launch()



