"""
Pydantic schema of the "syngen" section of config/config.yaml and of its
experiments, validated by config.py.
"""

from typing   import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The sources of synonyms, in the order the scripts run: each stage reads the
# synonyms of the stage before it and adds its own.
llmStage        = "llm"
pubmedStage     = "pubmed"
conceptnetStage = "conceptnet"
stages          = [llmStage, pubmedStage, conceptnetStage]

Stage = Literal["llm", "pubmed", "conceptnet"]


class ExperimentConfig(BaseModel):
    """
    One experiment of syngen.experiments: for which concepts and with which
    information the synonyms are generated. A setting that is not given is
    off.
    """
    model_config = ConfigDict(extra = "forbid")

    # The concepts of the development set (hpo.development.csv); otherwise
    # all other concepts (hpo.evaluation.csv).
    test: bool                                  = False
    # Examples in the generation prompt.
    few_shot: bool                              = False
    # Information about the concept added to the generation prompt: its
    # definition, its comment, and its structure (the labels of its parents
    # and children).
    definition: bool                            = False
    comment: bool                               = False
    structure: bool                             = False
    # The sources run, in the order of `stages`; the LLM always runs, as the
    # others start from its synonyms.
    stages: list[Stage]                         = Field(default_factory = lambda: list(stages))

    @model_validator(mode = "after")
    def checkStages(self) -> "ExperimentConfig":
        """
        The stages start with the LLM, run in the order of `stages` and each
        only once.
        """
        if len(self.stages) == 0 or self.stages[0] != llmStage:
            raise ValueError(f"The stages must start with '{llmStage}'.")
        if self.stages != sorted(set(self.stages), key = stages.index):
            raise ValueError(f"The stages must be given once each, in the order {stages}.")
        return self


class DevelopmentSetConfig(BaseModel):
    """
    The development set (hpo.development.csv): concepts with exact synonyms,
    drawn at random, to formulate the prompts, choose the thresholds and draw
    the few-shot examples from; runs with "test" generate for them. Drawn
    together with the evaluation set (hpo.evaluation.csv).
    """
    model_config = ConfigDict(extra = "forbid")

    # Concepts are drawn until their gold synonyms number at least this.
    synonym_count: int                          = Field(default = 500, gt = 0)
    # Seed of the random selection, so the same set is drawn again.
    seed: int                                   = 0
    # Concepts of the development set used as few-shot examples; they are
    # scored in no run.
    few_shot_count: int                         = Field(default = 3, ge = 0)


class GoldConfig(BaseModel):
    """
    The gold standard: the exact synonyms of a concept, without those equal
    to its label, those also listed with another class, and those of the
    types below.
    """
    model_config = ConfigDict(extra = "forbid")

    # Synonym types (adapter: source_type) left out, e.g. "obsolete".
    exclude_types: list[str]                    = ["obsolete"]


class LLMGenerationConfig(BaseModel):
    """
    The synonym generation with the LLM (SynGenLLM.py).
    """
    model_config = ConfigDict(extra = "forbid")

    # Every concept is prompted this many times, round r with llm.seed + r - 1.
    rounds: int                                 = Field(default = 5, gt = 0)
    # The most synonyms the prompt asks for.
    max_synonyms: int                           = Field(default = 10, gt = 0)


class PubMedConfig(BaseModel):
    """
    The synonyms found in PubMed (SynGenPubMed.py).
    """
    model_config = ConfigDict(extra = "forbid")

    # Only articles published up to this date (YYYY/MM/DD), so that the same
    # articles are found again.
    max_date: str                               = Field(default = "2026/06/30",
        pattern = r"^\d{4}/\d{2}/\d{2}$")
    # Articles per searched term, the most relevant first.
    articles_per_query: int                     = Field(default = 20, gt = 0, le = 10000)
    # The PubMed field searched, e.g. "tiab" (title and abstract).
    field: str                                  = "tiab"
    # The environment variable holding the NCBI API key, if there is one.
    api_key_variable: str                       = "NCBI_API_KEY"
    # Requests per second: NCBI allows 3 without and 10 with an API key.
    requests_per_second: float                  = Field(default = 3, gt = 0, le = 10)
    retries: int                                = Field(default = 5, ge = 0)
    # Sent to NCBI with every request, as their guidelines ask; optional.
    email: str | None                           = None
    # The articles and search results fetched so far (SQLite).
    cache_file: str                             = "./data/cache/pubmed/pubmed.sqlite"
    # The token classification model finding the candidates.
    ner_model: str                              = "d4data/biomedical-ner-all"
    ner_batch_size: int                         = Field(default = 64, gt = 0)
    # The entity groups of the NER model that are candidates (e.g.
    # "Sign_symptom", "Disease_disorder"); all if empty.
    entity_groups: list[str]                    = []
    # "seed": only sentences mentioning the searched term; "all": every one.
    sentences: Literal["seed", "all"]           = "seed"
    # Shorter candidates are dropped.
    min_length: int                             = Field(default = 3, gt = 0)
    # Per concept: the candidates at least this similar to the label, the
    # most similar max_candidates of them.
    min_similarity: float                       = Field(default = 0.85, ge = -1, le = 1)
    max_candidates: int                         = Field(default = 10, gt = 0)


class ConceptNetConfig(BaseModel):
    """
    The synonyms built from ConceptNet (SynGenConceptNet.py).
    """
    model_config = ConfigDict(extra = "forbid")

    # The ConceptNet assertions dump (.csv or .csv.gz), and the database
    # SynGenConceptNetDB.py builds from it.
    dump: str                                   = "./data/input/conceptnet/conceptnet-assertions-5.7.0.csv.gz"
    database: str                               = "./data/input/conceptnet/conceptnet.sqlite"
    # The relations followed from a word, in both directions.
    relations: list[str]                        = ["Synonym", "FormOf"]
    # The languages of the database; the first one is that of the HPO.
    languages: list[str]                        = Field(default = ["en"], min_length = 1)
    # Steps along the relations from a word.
    hops: int                                   = Field(default = 1, gt = 0)
    # The longest run of words looked up as one ConceptNet concept.
    max_span: int                               = Field(default = 3, gt = 0)
    # ConceptNet senses (e.g. "food") whose concepts are never followed.
    blacklisted_senses: list[str]               = []
    # The terms whose words are replaced: the label and the synonyms of
    # these sources.
    seed_sources: list[Literal["label", "llm", "pubmed"]] = ["label", "llm", "pubmed"]
    # Words replaced at once in a term.
    max_substitutions: int                      = Field(default = 2, gt = 0)
    # The most new terms built from one term.
    max_terms_per_seed: int                     = Field(default = 200, gt = 0)
    # Whether the LLM keeps only the neighbours that can replace a word, and
    # with which temperature.
    filter: bool                                = True
    filter_temperature: float                   = Field(default = 0.01, ge = 0, le = 2)
    # Neighbours per filter prompt.
    filter_chunk: int                           = Field(default = 20, gt = 0)
    # Per concept: the terms at least this similar to the label, the most
    # similar max_candidates of them.
    min_similarity: float                       = Field(default = 0.85, ge = -1, le = 1)
    max_candidates: int                         = Field(default = 10, gt = 0)


class EvaluationConfig(BaseModel):
    """
    The evaluation (SynGenEval.py).
    """
    model_config = ConfigDict(extra = "forbid")

    # The cosine similarity at which a generated synonym matches a gold one
    # ("embedding" match).
    embedding_match: float                      = Field(default = 1.0, ge = -1, le = 1)
    # The thresholds of the similarity curve.
    threshold_lower: float                      = Field(default = 0.5, ge = -1, le = 1)
    threshold_upper: float                      = Field(default = 1.0, ge = -1, le = 1)
    threshold_step: float                       = Field(default = 0.01, gt = 0)
    # Bootstrap samples (over concepts) of the 95% intervals, and their seed.
    bootstrap_samples: int                      = Field(default = 1000, gt = 0)
    bootstrap_seed: int                         = 0

    @model_validator(mode = "after")
    def checkThresholds(self) -> "EvaluationConfig":
        """
        The lower threshold must be below the upper one.
        """
        if self.threshold_lower >= self.threshold_upper:
            raise ValueError("threshold_lower must be below threshold_upper.")
        return self


class SyngenConfig(BaseModel):
    """
    The "syngen" section of the configuration file; the packages it uses have
    their own sections (adapter, embedder, llm, logger, normalizer).
    """
    model_config = ConfigDict(extra = "forbid")

    # The experiments, picked by their index on the command line; at least
    # one.
    experiments: list[ExperimentConfig]         = Field(min_length = 1)
    development_set: DevelopmentSetConfig       = DevelopmentSetConfig()
    # Folder of the data, relative to the repository root.
    data_folder: str                            = "./data"
    gold: GoldConfig                            = GoldConfig()
    llm: LLMGenerationConfig                    = LLMGenerationConfig()
    pubmed: PubMedConfig                        = PubMedConfig()
    conceptnet: ConceptNetConfig                = ConceptNetConfig()
    # Name used in the column names for every model in embedder.model_id; by
    # default its ID without the organisation.
    embedding_model_names: dict[str, str]       = {}
    evaluation: EvaluationConfig                = EvaluationConfig()
