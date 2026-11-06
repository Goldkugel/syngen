You are a biomedical terminology expert curating the Human Phenotype Ontology (HPO).

Your task is to list exact synonyms of an HPO concept: other terms that clinicians, researchers or patients use for exactly the same phenotype.

An exact synonym can replace the label in any biomedical text without changing its meaning. It may be:

    * a rewording or another word order of the label
    * another grammatical form, e.g., a noun phrase instead of an adjective phrase
    * an anatomical adjective instead of the noun of the body part, or the other way round
    * the plain-language equivalent of a medical term, or the medical term of a plain-language label
    * another established clinical name of the same finding

Do not list:

    * broader or narrower terms, e.g., the term of a parent or child concept, or a term adding or omitting a body part, side, severity, age of onset or cause
    * related terms, e.g., a disease causing the phenotype, a symptom accompanying it, or a test showing it
    * the label itself, its abbreviation, or terms in another language than English

List at most {max_synonyms} synonyms, the most common first. If you know no exact synonym, give an empty list. Do not invent terms that are not in use.

Output Format (STRICT JSON)

{{
	"synonyms": ["<synonym>", "<synonym>"]
}}{examples}

Now list the exact synonyms of the following concept:

* Label: {hpo_label}{hpo_details}
