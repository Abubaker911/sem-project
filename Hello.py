import streamlit as st
from transformers import pipeline
from sklearn.metrics import accuracy_score
from tune_sklearn import TuneSearchCV

# Load the spell checker model
spell_checker = pipeline(task="fill-mask", model="bert-base-uncased")

# Sample data
text_samples = [
    "I likke to reaad books.",
    "Thiss is a speling cheker.",
    "The quick brown fox jumped over the lazy dog.",
]

# Function to correct spelling
def correct_spelling(text):
    # Check if [MASK] token is present in the input text
    if "[MASK]" not in text:
        # If not present, add [MASK] token to the end of the text
        text += " [MASK]"

    # Use the fill-mask pipeline to correct spelling
    return spell_checker(text)[0]['sequence'].strip()

# Streamlit UI
st.title("Spell Checker")

text_input = st.text_area("Enter text:")
if st.button("Check Spelling"):
    corrected_text = correct_spelling(text_input)
    st.markdown(f"**Corrected Text:** {corrected_text}")

# Tune hyperparameters
if st.checkbox("Tune Hyperparameters"):
    labeled_data = [
        ("I like to read books.", "I likke to reaad books."),
        ("This is a spelling checker.", "Thiss is a speling cheker."),
        (
            "The quick brown fox jumped over the lazy dog.",
            "The quick brown fox jumped over the lazy dog.",
        ),
    ]

    X_train, y_train = zip(*labeled_data)

    # Define the hyperparameter search space
    param_space = {
        "model": ["bert-base-uncased", "bert-large-uncased"],
        "task": ["fill-mask", "text-generation"],
    }

    # Define the model to be tuned
    tuned_model = pipeline(task="fill-mask", model="bert-base-uncased")

    # Use TuneSearchCV for hyperparameter tuning
    search = TuneSearchCV(tuned_model, param_space, n_jobs=-1, scoring="accuracy", cv=2)
    search.fit(X_train, y_train)

    # Display best hyperparameters and accuracy
    st.markdown("**Best Hyperparameters:**")
    st.write(search.best_params_)
    st.markdown(f"**Best Accuracy:** {search.best_score_}")

# Display sample data
st.markdown("**Sample Data:**")
for sample in text_samples:
    st.write(sample)

