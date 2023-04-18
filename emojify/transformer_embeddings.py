import sentence_transformers as st

model = st.SentenceTransformer("all-MiniLM-L6-v2")

# Our sentences we like to encode
sentences = [
    "This framework generates embeddings for each input sentence",
    "Sentences are passed as a list of string.",
    "The quick brown fox jumps over the lazy dog.",
]


def get_transformer_embeddings(sentences):
    return model.encode(sentences)


def main():
    print("get_transformer_embeddings")
    some = [get_transformer_embeddings(s) for s in sentences]
    print("single embeddings", some[0])
    print("single embeddings size", some[0].size)  # 348
    print("all")
    print(some)


if __name__ == "__main__":
    main()
