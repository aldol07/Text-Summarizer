import numpy as np

from textSummarizer.extractive.base import ExtractiveSummarizer
from textSummarizer.extractive.vectorizer import TfidfVectorizer
from textSummarizer.preprocessing import Document, Preprocessor

WEIGHTINGS = ("tfidf", "binary")


class LsaSummarizer(ExtractiveSummarizer):
    """Latent Semantic Analysis summarization (Steinberger & Ježek, 2004).

    Factorizes the term-by-sentence matrix ``A = U Σ Vᵀ``. Each singular
    vector is a latent topic, and ``Vᵀ[t, i]`` is how strongly sentence i
    expresses topic t. A sentence's score is its length in the reduced topic
    space::

        score(i) = sqrt( Σ_t (σ_t · Vᵀ[t, i])² )

    Uses the top ``n_topics`` topics, or every topic with
    ``σ_t >= sigma_threshold · σ_max`` when ``n_topics`` is None.

    Defaults (binary weighting, 2 topics) were picked on the CNN/DailyMail
    *validation* split (ROUGE-1 33.7 vs 29.2 for TF-IDF with all strong topics),
    never on the test split.

    Sentence columns are deliberately *not* length-normalized. With unit
    columns, every sentence has the same total mass, so a four-word sentence
    aligned with one topic outranks a rich 30-word one. On CNN/DailyMail that
    dropped ROUGE-1 below the random baseline.
    """

    name = "lsa"
    description = "Topic strength in the SVD latent space (Steinberger & Ježek, 2004)"

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        n_topics: int | None = 2,
        sigma_threshold: float = 0.5,
        weighting: str = "binary",
    ):
        super().__init__(preprocessor)
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")
        if n_topics is not None and n_topics < 1:
            raise ValueError("n_topics must be >= 1")
        self.n_topics = n_topics
        self.sigma_threshold = sigma_threshold
        self.weighting = weighting

    def score_sentences(self, doc: Document) -> np.ndarray:
        if self.weighting == "binary":
            vectorizer = TfidfVectorizer(binary=True, use_idf=False, normalize=False)
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, normalize=False)
        term_sentence = vectorizer.fit_transform(doc.term_lists).matrix.T
        if term_sentence.size == 0 or not term_sentence.any():
            return np.zeros(len(doc))

        _, sigma, vt = np.linalg.svd(term_sentence, full_matrices=False)
        if self.n_topics is not None:
            k = min(self.n_topics, sigma.size)
        else:
            k = max(1, int((sigma >= self.sigma_threshold * sigma[0]).sum()))
        return np.sqrt(((sigma[:k, None] * vt[:k]) ** 2).sum(axis=0))
