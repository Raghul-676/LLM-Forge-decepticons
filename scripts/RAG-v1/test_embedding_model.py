from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    print("=" * 70)
    print("RAG PROTOTYPE V1 - EMBEDDING TEST")
    print("=" * 70)

    print("Loading embedding model:")
    print(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    documents = [
        (
            "constitution_article_21",
            "No person shall be deprived of his life or personal "
            "liberty except according to procedure established by law."
        ),
        (
            "tax_example",
            "Goods and Services Tax is imposed on the supply of "
            "goods and services."
        ),
        (
            "property_example",
            "This provision concerns transfer and ownership of property."
        ),
        (
            "criminal_example",
            "A person accused of an offence may have procedural "
            "protections under criminal law."
        ),
    ]

    query = (
        "What protection does the Constitution give to "
        "a person's life and personal liberty?"
    )

    print("\nQuery:")
    print(query)

    document_texts = [
        text for _, text in documents
    ]

    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Because vectors are normalized, dot product
    # is equivalent to cosine similarity.
    scores = np.dot(
        document_embeddings,
        query_embedding
    )

    ranking = np.argsort(scores)[::-1]

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for rank, index in enumerate(ranking, start=1):
        document_id, text = documents[index]

        print(
            f"\nRank {rank}"
            f"\nScore: {scores[index]:.4f}"
            f"\nID: {document_id}"
            f"\nText: {text}"
        )

    best_id = documents[ranking[0]][0]

    print("\n" + "=" * 70)

    if best_id == "constitution_article_21":
        print("EMBEDDING TEST PASSED")
    else:
        print("EMBEDDING TEST NEEDS REVIEW")

    print("=" * 70)


if __name__ == "__main__":
    main()