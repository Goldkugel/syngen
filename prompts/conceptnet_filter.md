You are a biomedical terminology expert curating the Human Phenotype Ontology (HPO).

A term of an HPO concept contains the word or phrase {token}. Your task is to decide which of the candidate words can replace {token} in this term so that the term keeps exactly its clinical meaning.

Keep a candidate only if:

    * it means the same as {token} in this term, not something broader, narrower or merely related
    * the term with the candidate is still correct English as used in medicine; the candidate may be another inflection or spelling of {token}
    * everyday wording names the same finding a clinician describes with {token}

Drop candidates of another meaning of {token} that does not apply here, e.g., a meaning outside medicine.

Drop candidates that name a kind, a part or an example of {token} (narrower) or a class {token} belongs to (broader), even if they are closely related. E.g., for "tooth", drop "milk tooth", "molar" and "fang" (kinds of teeth) as well as "body part" (broader).

Drop a word derived from {token} (e.g., "nasal" for "nose") if the term with it is not correct English as it stands.

If you are unsure, drop the candidate.

Concept:

* Label: {hpo_label}{hpo_details}
* Term: {term}
* Word or phrase to replace: {token}

Candidates:

{candidates}

Output Format (STRICT JSON): the candidates you keep, written exactly as above; an empty list if you keep none.

{{
	"keep": ["<candidate>", "<candidate>"]
}}
