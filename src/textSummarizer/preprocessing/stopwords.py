"""Built-in English stopword list.

Mirrors NLTK's English list so the engine works without downloading NLTK
corpora. Tokens are split on apostrophes, so contractions appear as their
fragments ("don", "t", "ll", ...).
"""

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself yourselves
    he him his himself she her hers herself it its itself they them their
    theirs themselves what which who whom this that these those am is are was
    were be been being have has had having do does did doing a an the and but
    if or because as until while of at by for with about against between into
    through during before after above below to from up down in out on off over
    under again further then once here there when where why how all any both
    each few more most other some such no nor not only own same so than too
    very s t can will just don should now d ll m o re ve y ain aren couldn
    didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn
    weren won wouldn
    """.split()
)

# Extra function words and reporting verbs that make poor keywords
# ("said", "would", "also"). Used only by keyword extraction, so summarization
# stays comparable with other NLTK-based implementations.
KEYWORD_STOPWORDS: frozenset[str] = ENGLISH_STOPWORDS | frozenset(
    """
    would could might must shall may also said says say saying told tell
    according one two three many much several however yet still even though
    although well like get got make made us within without upon whether
    whose every another new including include includes
    hey hi hello ok okay yeah yes yep nope sure thanks thank lol haha hmm oh please
    """.split()
)
