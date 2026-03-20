import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from config import *

def semanticClassificationPrompt1(
        label : str, 
        definition : str, 
        comment : str,  
        parents : list,
        children : list
) -> str:
    return \
    "You are a biomedical ontology curator working with the Human Phenotype " \
        "Ontology (HPO).\n\n" \
    "Please, interpret the meaning and scope of the following HPO concept " \
        "using the provided ontology information.\n\n" \
    f"Label: {quote(label)}\n" \
    f"Definition: {quote(definition)}\n" \
    f"Parent Concept Labels: {applyFormat(parents)}\n" \
    f"Child Concept Labels: {applyFormat(children)}\n" \
    f"Comment: {quote(comment)}\n\n" \
    "Explain:\n\n" \
    "1. The phenotype described by the concept.\n" \
    "2. The scope of the concept based on the ontology hierarchy."

def semanticClassificationPrompt2(
    synonym : str
) -> str:
    return \
    "Please, interpret the biomedical meaning of the following term.\n\n" \
    f"Term: {quote(synonym)}\n" \
    "Explain what phenotype, condition, or clinical " \
        "description this term usually refers to in biomedical or " \
        "clinical terminology."

def semanticClassificationPrompt3() -> str:
    return \
    "Now, please assist with ontology curation in the Human Phenotype " \
        "Ontology (HPO).\n\n" \
    "Compare the meaning of the HPO concept and the term that I gave you.\n\n" \
    "Please explain how the term relates to the concept.\n\n" \
    "Consider whether the term:\n\n" \
    "- has identical meaning\n" \
    "- has a broader meaning than the concept\n" \
    "- has a narrower meaning than the concept\n" \
    "- refers to a related condition\n" \
    "- describes a cause or consequence\n" \
    "- omits important aspects of the concept definition\n\n" \
    "Important scope check:\n\n" \
    "Determine whether the term captures the full phenotype described by " \
        "the concept definition.\n\n" \
    "Specifically check whether the term:\n\n" \
    "- omits key characteristics of the concept\n" \
    "- describes only a general manifestation of the concept\n" \
    "- represents a broader clinical description\n" \
    "- represents a more specific subtype of the concept\n\n" \
    "If the term omits defining features or represents a broader or narrower " \
        "phenotype, it should not be considered semantically identical to " \
        "the concept."

def semanticClassificationPrompt4(fewShot : bool = False) -> str:
    ret = \
    "Finally, classify the term according to the official HPO synonym " \
        "definitions.\n\n" \
    "HPO synonym definitions:\n\n" \
    "\"Exact\": Exact synonyms can be used interchangeably. One synonym " \
        "can replace the other without changing the meaning. Example: " \
        "\"Focal myoclonic seizure\" and \"Partial myoclonic seizure\".\n" \
    "\"Related\": Related synonyms are conceptually related but not exactly " \
        "interchangeable. One synonym does not replace the other. Example: " \
        "\"Myocardial infarction\" and \"Coronary artery disease\". " \
        "\"Coronary artery disease\" is an underlying condition that may " \
        "lead to, but does not necessarily imply, a \"myocardial " \
        "infarction\".\n\n" \
    "Important ontology synonym rules:\n\n" \
    "In biomedical ontologies, lexical similarity does NOT automatically " \
        "imply an Exact synonym.\n\n" \
    "The following types of terms should usually be classified as " \
        "\"Related\":\n\n" \
    "- Abbreviations or short forms\n" \
    "- Chemical formulas or structural formulas\n" \
    "- Systematic chemical names vs common names\n" \
    "- Translations into other languages\n" \
    "- Singular vs plural variants\n" \
    "- Identifiers or codes\n\n" \
    "These should be classified as Related unless the synonym is literally " \
        "interchangeable in normal biomedical text.\n\n" \
    "Decision rules:\n\n" \
    "- identical meaning → Exact\n" \
    "- broader or narrower → Related\n" \
    "- cause, consequence, or associated condition → Related\n" \
    "- partial overlap or uncertainty → Related\n\n" \
    "Before deciding, ask yourself:\n\n" \
    "Would replacing the concept label with the synonym always produce a " \
        "natural biomedical sentence?\n\n" \
    "If the substitution would look unusual, abbreviated, overly technical, " \
        "or language-specific, classify it as \"Related\".\n\n"
    
    if fewShot:
        ret = ret + "Examples:\n\n" \
            "- The term \"calcifediolum\" is classified as \"Related\" to " \
                "the concept with the label \"calcidiol\".\n" \
            "- The term \"2,5-diaminopentanoic acid\" is classified as " \
                "\"Exact\" to the concept with the label \"ornithine\".\n" \
            "- The term \"double negative memory B-lymphocyte\" is " \
                "classified as \"Exact\" to the concept with the label " \
                "\"double negative memory B cell\".\n" \
            "- The term \"upregulation of membrane invagination\" is " \
                "classified as \"Exact\" to the concept with the label " \
                "\"positive regulation of membrane invagination\".\n" \
            "- The term \"Hypotrophic maxilla\" is classified as \"Related\" " \
                "to the concept with the label \"Hypoplasia of the " \
                "maxilla\".\n" \
            "- The term \"Upper jaw retrusion\" is classified as \"Exact\" " \
                "to the concept with the label \"Hypoplasia of the " \
                "maxilla\".\n" \
            "- The term \"fore epipodium\" is classified as \"Related\" to " \
                "the concept with the label \"forelimb zeugopod\".\n" \
            "- The term \"5th digit of hand\" is classified as \"Exact\" to " \
                "the concept with the label \"manual digit 5\".\n\n"
    
    ret = ret + "Please answer only with a valid JSON object containing:\n\n" \
        "{\n" \
        f"{quote(synonymClass)}: \"Exact\" | \"Related\",\n" \
        f"{quote(confidenceColumn)}: <number between 0 and 10>\n" \
        "}"

    return ret

def semanticClassificationPrompt(
        label : str, 
        definition : str, 
        comment : str, 
        parents : list, 
        children : list,
        synonym : str
)-> str:
    return f"""
You are an expert in biomedical terminology and ontologies. 

Your task is to decide whether the given synonym is an "Exact" synonym or a "Related" synonym of the concept label.

Definitions:

- Exact: The synonym describes exactly the same phenotype and can replace the concept label in medical text without changing the meaning, e.g., "Focal myoclonic seizure" vs. "Partial myoclonic seizure".
- Related: The synonym is associated with the concept but refers to a different concept, a broader class, a narrower class, or a commonly associated entity, e.g., "Myocardial infarction" vs. "Coronary artery disease".

Important ontology rules:

Classify as "Related" if the synonym is:
- a chemical formula (e.g., H2N-CH2-COOH)
- a chemical systematic name
- a chemical identifier or registry name
- a molecular abbreviation (e.g., Gly, ALA, 5-HT)
- a gene or protein symbol
- a short acronym or abbreviation
- a plural or grammatical variant (e.g., ion vs ions)
- a Latin anatomical name (e.g., nodus lymphaticus)
- a broader or more generic term
- a narrower subtype
- a commonly associated condition
- a cause or consequence of the phenotype

Classify as "Exact" if the synonym:
- describes the same phenotype using different wording
- is a clinical paraphrase with identical meaning
- is a reordered phrase with the same meaning
- replaces words with equivalent medical terms (e.g., hypoplasia vs underdevelopment)
- describes the same anatomical abnormality

Information regarding the HPO concept:

Concept label: {quote(label)}
Definition: {quote(definition)}
Parent concept labels: {applyFormat(parents)}
Child concept labels: {applyFormat(children)}
Comment (may be empty): {quote(comment)}

Information regarding the synonym to classify:

Synonym to classify: {quote(synonym)}

Important: 

Please answer only with a valid JSON object containing:\n\n""" \
"{\n" \
f"{quote(synonymClass)}: \"Exact\" | \"Related\",\n" \
f"{quote(confidenceColumn)}: <number between 0 and 10>\n" \
"}"





















































def createExampleString(examples : list, exStr : str = "Example") -> str:
    ret = "None"
    if len(examples) > 0:
        ret = ""
        for index, example in enumerate(examples):
            ret += f"{exStr} {index+1}: " + applyFormat(example) + "\n"
        ret = ret[0:len(ret) - 1]

    return ret

def quote(string : str) -> str:
    return quotationCharacter + string + quotationCharacter

def applyFormat(l : list[str] = []) -> str:
    string = "None"
    if len(l) > 0:
        if isinstance(l, str):
            string = quote(l)
        else:
            string = quote((quotationCharacter + ", " + 
                quotationCharacter).join(l))
    return string

def getPreTaskSystem() -> str:
    return f"You are a biomedical ontology expert with expertise in " \
        "clinical phenotype terminology and precise medical condition " \
        "definitions."

def getPreTaskPart1(
    label : str
) -> str:
    return \
    "Let us create a medically precise definition of the " \
        "following concept of a medical condition.\n\n" \
    "The medical condition is: {}.\n\n" \
    "Based on your understanding of the condition, generate a clear and " \
        "medically precise definition of the term.".format(
        quote(label)
    )

def getPreTaskPart2(parents : list) -> str:
    return \
    "This is the list of parent concept(s) of the medical condition:\n\n" \
    "Parent Concept(s): {}.\n\n" \
    "Does the definition describe a condition that is fully consistent " \
        "with — and more specific than — each of its parent concepts?".format(
        applyFormat(parents) 
    )

def getPreTaskPart3(children : list) -> str:
    return \
    "This is the list of child concept(s) of the medical condition:\n\n" \
    "Child Concept(s): {}.\n\n" \
    "Does the definition describe something that includes all of the child " \
        "concept(s) as specific instances, without being so narrow that " \
        "any child would be left out?".format(
        applyFormat(children) 
    )

def getPreTaskPart4() -> str:
    return \
    "Provide only the final, validated definition of the medical " \
        "condition.\n\n" \
    "Your response must consist of a single, medically accurate sentence " \
        "that defines the condition.\n\n" \
    "Do not include any introductory phrases, commentary, bullet points, " \
        "or additional information. Output only the definition text — " \
        "nothing else."

def getAlternativeComplexPrompt1(
        label : str, 
        definition : str, 
        comment : str, 
        parents : list = [], 
        children : list = []
) -> str:
    return "You are an ontology-focused biomedical language model tasked " \
        "with generating EXACT synonyms for a Human Phenotype Ontology " \
        "(HPO) term.\n\n" \
    "Below is the information about the target HPO term.\n\n" \
    f"Label]: {quote(label)}\n" \
    f"Definition(s): {quote(definition)}\n" \
    f"Comment: {quote(comment)}\n" \
    f"Parent Concept(s): {applyFormat(parents)}\n" \
    f"child Concept(s): {applyFormat(children)}\n\n" \
    "You may use the parent and child terms only to understand the semantic" \
        "boundaries of the target concept. Do not generate synonyms for the " \
        "child or parent terms, and do not use any wording that corresponds " \
        "specifically to a child or parent term.\n\n" \
    "Your task is to generate ONLY *exact* synonyms that are strictly " \
    "interchangeable with the provided HPO label. These synonyms must:\n\n" \
    "• Preserve identical meaning without adding, removing, or modifying " \
        "any semantic component of the phenotype.\n" \
    "• Remain at the same level of granularity as the original term.\n" \
    "• Avoid implying a cause, mechanism, diagnosis, severity, " \
        "timeframe, or anatomical shift not present in the definition.\n" \
    "• Represent standard biomedical phrasing, not colloquial language.\n" \
    "• Not be broader, narrower, related-but-not-equivalent, or figurative.\n" \
    "• Not overlap with or drift toward the meaning of any child or parent " \
        "term.\n\n" \
    "Output only synonyms that can be used in exactly the same contexts " \
        "as the original label in phenotype annotation."

def getAlternativeComplexPrompt2() -> str:
    return \
    "You will now evaluate all candidate synonyms generated so far.\n\n" \
    "Your task is to remove every candidate that fails strict " \
        "equivalence criteria.\n\n" \
    "A candidate MUST be eliminated if it does any of the following:\n\n" \
    "— SEMANTIC DRIFT —\n" \
    "• Broadens or narrows the meaning relative to the original label.\n" \
    "• Introduces or removes any semantic component.\n" \
    "• Overlaps with, resembles, or implies the meaning of any child term.\n" \
    "• Implies abnormality in a different structure or anatomical region.\n\n" \
    "— CLINICAL OR CAUSAL IMPLICATIONS —\n" \
    "• Suggests a cause, mechanism, etiology, risk factor, or diagnostic" \
        "process.\n\n" \
    "— NON-PHENOTYPIC OR NON-BIOMEDICAL LANGUAGE —\n" \
    "• Uses colloquial, figurative, idiomatic, or ambiguous terminology.\n" \
    "• Uses outdated, rarely used, or unstandardized phrasing in clinical " \
        "practice.\n\n" \
    "— LINGUISTIC OR STRUCTURAL ISSUES —\n" \
    "• Rephrases the term in a way that alters focus or specificity.\n" \
    "• Breaks biomedical naming conventions.\n" \
    "• Collapses separate semantic components into one, or splits them " \
        "apart.\n\n" \
    "Only retain candidates that:\n" \
    "• are strictly equivalent in meaning to the original HPO label,\n" \
    "• preserve its full semantic scope and boundaries,\n" \
    "• adhere to standard biomedical phrasing,\n" \
    "• and can be used interchangeably in phenotype annotation.\n\n" \
    "Output the curated list of surviving candidates."

def getAlternativeComplexPrompt3() -> str:
    return \
    "Before generating final list of candidate synonyms, create controlled " \
        "rephrasings of the surviving candidates and the label. These " \
            "rephrasings must:\n\n" \
    "• Preserve strict semantic equivalence with the surviving candidates or " \
        "the label.\n" \
    "• Maintain the same level of specificity and anatomical/phenotypic " \
        "focus.\n" \
    "• Follow standard biomedical phrasing conventions.\n" \
    "• Avoid introducing causality, severity, temporality, " \
        "diagnostic implications, or any expansion/reduction of meaning.\n" \
    "• Avoid wording that corresponds to any child or parent term or " \
        "implies one of them.\n\n" \
    "Generate rephrasings by transforming the label and candidate synonyms " \
        "through conventional biomedical linguistic operations, such as:\n\n" \
    "• Switching between noun phrase and adjectival constructions " \
        "(e.g., “Pubertal delay” → “Delayed puberty”).\n" \
    "• Reordering components without changing meaning.\n" \
    "• Using standard nominalized or adjectival variants.\n" \
    "• Converting passive ↔ active or descriptive forms in medically " \
        "accepted ways.\n" \
    "• Producing equivalent forms that keep all semantic components intact.\n" \
    "All rephrased forms must be fully interchangeable with the original " \
        "label or candidate synonym in phenotype annotation.\n" \
    "Output the new curated list of candidate synonyms containing also the " \
        "rephrased terms."

def getAlternativeComplexPrompt4() -> str:
    return \
    "You now have a curated list of synonym candidates that survived the " \
        "filtering process. Before finalizing them, perform a strict " \
        "consistency evaluation to ensure that every remaining candidate " \
        "is a true EXACT synonym of the original HPO label.\n\n" \
    "For EACH remaining candidate, verify all of the following:\n\n" \
    "— SEMANTIC CONSISTENCY —\n" \
    "• It preserves the full meaning of the original HPO label with " \
        "no additions, reductions, shifts in scope, or changes in " \
        "anatomical focus.\n" \
    "• It aligns precisely with the definition and logical " \
        "constraints provided.\n" \
    "• It does not overlap with or drift toward the meaning of any child " \
        "or parent term.\n\n" \
    "— LINGUISTIC CONSISTENCY —\n" \
    "• It represents standard biomedical phrasing used in clinical or " \
        "scientific contexts.\n" \
    "• It does not introduce ambiguity, idiomatic language, colloquial " \
        "terms, or unusual constructions.\n" \
    "• It maintains the same level of granularity as the original term " \
        "and follows typical ontology naming conventions.\n\n" \
    "— CONTEXTUAL CONSISTENCY —\n" \
    "• It remains interchangeable with the original label in " \
        "phenotype annotation.\n" \
    "• It can be used without altering interpretation in any context where " \
        "the original label is appropriate.\n" \
    "• It does not imply any clinical workflow, causality, severity, or " \
        "temporal dimension not present in the original definition.\n\n" \
    "If a candidate fails ANY of the criteria above, remove it.\n" \
    "After performing this final review, output ONLY the candidates that " \
        "satisfy all consistency requirements."

def getAlternativeComplexPrompt5() -> str:
    return \
    "You will now prepare the surviving candidate synonyms for final " \
        "evaluation by placing them into a strict, machine-readable output " \
        "structure.\n\n" \
    "Follow these rules:\n\n" \
    "• Do NOT include any chain-of-thought or internal reasoning.\n" \
    "• Do NOT describe your decision process.\n" \
    "• Include ONLY the elements explicitly requested below.\n" \
    "• Do NOT invent additional fields.\n" \
    "Use the following exact JSON structure:\n\n" \
    "{\n" \
    "\t\"exact_synonyms\": [\n" \
    "\t\t\"<<list the remaining exact synonyms>>\"" \
    "\t]\n"\
    "}\n\n" \
    "EXAMPLE OUTPUT (for demonstration only):\n\n" \
    "{\n" \
    "\t\"exact_synonyms\": [\n" \
    "\t\t\"Delayed puberty\",\n" \
    "\t\t\"Pubertal delay\",\n" \
    "\t\t\"Late onset of puberty\"\n" \
    "\t]\n" \
    "}\n\n" \
    "Produce exactly one JSON object as specified above."

def getSynonymClassPrompt(
        label : str, 
        definition : str, 
        comment : str, 
        parents : list, 
        children : list,
        synonym : str
)-> str:
    return f"""
You are an expert in biomedical terminology and ontologies. 

Your task is to decide whether the given synonym is an "Exact" synonym or a "Related" synonym of the concept label.

Definitions:

- Exact: The synonym describes exactly the same phenotype and can replace the concept label in medical text without changing the meaning, e.g., "Focal myoclonic seizure" vs. "Partial myoclonic seizure".
- Related: The synonym is associated with the concept but refers to a different concept, a broader class, a narrower class, or a commonly associated entity, e.g., "Myocardial infarction" vs. "Coronary artery disease".

Important ontology rules:

Classify as "Related" if the synonym is:
- a chemical formula (e.g., H2N-CH2-COOH)
- a chemical systematic name
- a chemical identifier or registry name
- a molecular abbreviation (e.g., Gly, ALA, 5-HT)
- a gene or protein symbol
- a short acronym or abbreviation
- a plural or grammatical variant (e.g., ion vs ions)
- a Latin anatomical name (e.g., nodus lymphaticus)
- a broader or more generic term
- a narrower subtype
- a commonly associated condition
- a cause or consequence of the phenotype

Classify as "Exact" if the synonym:
- describes the same phenotype using different wording
- is a clinical paraphrase with identical meaning
- is a reordered phrase with the same meaning
- replaces words with equivalent medical terms (e.g., hypoplasia vs underdevelopment)
- describes the same anatomical abnormality

Information regarding the HPO concept:

Concept label: {quote(label)}
Definition: {quote(definition)}
Parent concept labels: {applyFormat(parents)}
Child concept labels: {applyFormat(children)}
Comment (may be empty): {quote(comment)}

Information regarding the synonym to classify:

Synonym to classify: {quote(synonym)}

Important: 

- Return only one word as the output: "Exact" or "Related".
"""

"""
    "Your Task:\n\n" \
    "You are given a primary Human Phenotype Ontology (HPO) concept and a " \
        "candidate synonym. Using only the information provided and HPO " \
        "curation conventions, classify the synonym into exactly one of " \
        "the following classes:\n\n" \
    "\"Exact\", \"Broad\", \"Narrow\", or \"Related\"\n\n" \
    "Your goal is to decide whether the candidate synonym could be used " \
        "interchangeably with the primary label in phenotype annotation, " \
        "not whether it is clinically, mechanistically, or etiologically " \
        "identical.\n\n" \
    "Mandatory ontology rules:\n\n" \
    "Apply all of the following rules:\n" \
    "- Collapse etiology, timing, and mechanism: Ignore differences such as " \
        "congenital vs acquired, mechanism vs manifestation, histology vs " \
        "appearance, or cause vs effect unless the synonym explicitly " \
        "excludes part of the label’s meaning.\n" \
    "- Treat historical, pathological, and descriptive disease names as " \
        "\"Exact\": If multiple names refer to the same recognized disease " \
        "entity (including deprecated, histologic, or descriptive names), " \
        "classify as \"Exact\", even if they emphasize different features.\n" \
    "- Accept lay, colloquial, and paraphrased descriptions as \"Exact\": " \
        "Plain-language descriptions, shorthand phrases, or less technical " \
        "wording are Exact if they describe the same observable phenotype.\n" \
    "- Ignore pluralization and count: Singular vs plural forms (e.g., " \
        "\"tumor\" vs \"tumors\", \"head\" vs \"heads\") are lexical " \
        "variants, not semantic changes.\n" \
    "- Named clinical signs ≡ defining descriptions: A named sign (e.g., " \
        "\"sandal gap\", \"prognathia\") and a phrase that directly " \
        "describes it are Exact, even if the description sounds broader or " \
        "vaguer.\n" \
    "- Size, projection, prominence, and excess are interchangeable in " \
        "dysmorphology: Terms such as large, big, enlarged, prominent, " \
        "projecting, excess, hyperplasia are \"Exact\" when they refer to " \
        "the same anatomical structure and direction of change.\n" \
    "- Anatomical shorthand is acceptable: Closely related anatomical terms " \
        "commonly used interchangeably in phenotype annotation (e.g., " \
        "\"jaw\" ↔ \"mandible\", \"nasal ridge\" ↔ \"nasal dorsum\") are " \
        "\"Exact\" unless a clear exclusion is stated.\n" \
    "- Laboratory proxies and functional readouts may be \"Exact\": If a " \
        "laboratory measurement or functional descriptor is the standard " \
        "phenotypic manifestation of the label, classify as \"Exact\" even " \
        "if it represents a mechanism.\n\n" \
    "When NOT to use \"Exact\":\n\n" \
    "Only choose \"Broad\", \"Narrow\", or \"Related\" if one of the " \
        "following is clearly true:\n\n" \
    "- \"Broad\": The synonym explicitly includes additional phenotypes " \
        "or anatomical regions not covered by the label.\n" \
    "- \"Narrow\": The synonym explicitly refers to a subset or subtype " \
        "of the label that does not cover all instances.\n" \
    "- Related: The synonym describes a different pathological process, " \
        "disease category, or downstream condition that cannot replace the " \
        "label in annotation.\n\n" \
    "Do not downgrade to \"Broad\"/\"Narrow\"/\"Related\" solely because " \
        "of:\n\n"  \
    "- Different terminology\n- Different emphasis\n- Added descriptive " \
        "detail\n- Clinical causality\n- Pathological mechanism\n" \
        "- Severity or degree\n- Common clinical usage differences\n\n" \
    "Primary HPO concept:\n\n" \
    f"- Label: {quote(label)}\n" \
    f"- Definition: {quote(definition)}\n" \
    f"- Comment: {quote(comment)}\n" \
    f"- Parent concept(s): {applyFormat(parents)}\n" \
    f"- Child concept(s): {applyFormat(children)}\n\n" \
    "Candidate synonym:\n\n" \
    f"- Synonym: {quote(synonym)}\n\n" \
    "Class definitions:\n\n" \
    "- \"Exact\": Interchangeable in phenotype annotation.\n" \
    "- \"Broad\": More general than the label.\n" \
    "- \"Narrow\": More specific than the label.\n" \
    "- \"Related\": Conceptually associated but not interchangeable.\n\n" \
    "Output format (strict):\n\n" \
    "Output exactly one of the following and nothing else: \"Exact\", " \
        "\"Broad\", \"Narrow\", and \"Related\"\n." \
    "Do not include explanations or reasoning."
"""

def getSynonymTypePrompt(
        label : str, 
        definition : str, 
        comment : str,  
        parents : list,
        children : list, 
        synonym : str
)-> str:
    return \
    "Your Task:\n\n" \
    "You are given information about a primary Human Phenotype Ontology " \
        "(HPO) concept and a candidate synonym.\n\n" \
    "Your task is to classify the candidate synonym according to its " \
        "language register, not its semantic relationship to the primary " \
        "concept.\n\n" \
    "Important:\n\n" \
    "This is NOT a task about whether the synonym is a direct match, " \
        "broader term, narrower term, or spelling variant.\n" \
    "The ONLY goal is " \
        "to determine whether the synonym is written in lay (everyday) " \
        "language or expert (professional medical) language.\n" \
    "Use only the information provided.\n\n" \
   "Primary HPO concept:\n\n" \
    f"- Label: {quote(label)}\n" \
    f"- Definition: {quote(definition)}\n" \
    f"- Comment: {quote(comment)}\n" \
    f"- Parent concept(s): {applyFormat(parents)}\n" \
    f"- Child concept(s): {applyFormat(children)}\n\n" \
    "Candidate synonym:\n\n" \
    f"- Label: {quote(synonym)}\n\n" \
    "Class definitions:\n\n" \
    "- Layperson: Uses everyday, non-technical language that would be " \
        "understood by the general public. These expressions avoid " \
        "specialized medical jargon and often describe conditions in " \
        "plain English. (e.g., 'heart attack' instead of 'myocardial " \
        "infarction', 'tooth decay' instead of 'dental caries', 'small " \
        "lung' instead of 'pulmonary hypoplasia').\n" \
    "- Expert: Uses technical, clinical, anatomical, Greek/Latin-derived, " \
        "or standardized medical terminology intended for communication " \
        "among healthcare professionals or researchers (e.g., " \
        "'myocardial infarction', 'hemarthrosis', 'fetal hypokinesia', " \
        "'pulmonary hypoplasia').\n\n" \
    "Decision rules:\n\n" \
    "- If medical terminology is replaced with plain English, classify " \
        "as Layperson.\n" \
    "- If the synonym contains formal diagnostic, pathological, or " \
        "anatomical terminology, classify as Expert.\n" \
    "- Plain-English anatomical descriptions (e.g.,'middle finger bone', " \
        "'white patch in the mouth', 'underdeveloped lung') should be " \
        "classified as Layperson, even if medically accurate.\n" \
    "- Classify based on the dominant wording of the synonym.\n" \
    "- Use only the information provided.\n\n" \
    "Output instructions:\n\n" \
    f"Do not include explanations or any additional text. Output exactly " \
        f"one word: '{laypersonSynonymType.capitalize()}' or " \
        f"'{expertSynonymType}'."

"""
    "Your Task:\n\n" \
    "You are given information about a primary Human Phenotype Ontology " \
        "(HPO) concept and a candidate synonym.\n\n" \
    "Your task is to classify the candidate synonym according to its " \
        "language register ONLY — not its semantic correctness and not its " \
        "relationship to the primary label.\n\n" \
    "IMPORTANT:\n\n" \
    "This is strictly a linguistic register task.\n" \
    "The fact that a term refers to a medical condition does NOT make " \
        "it \"Expert\".\n" \
    "Classify based on wording style, not subject matter.\n" \
    "Ignore the wording of the primary HPO label when determining register.\n" \
    "Evaluate ONLY the candidate synonym.\n\n" \
    "CORE PRINCIPLE:\n\n" \
    "\"Expert\": specialist medical terminology intended for professional " \
        "communication.\n" \
    "\"Layperson\": plain English wording that could naturally appear in " \
        "patient-facing or general public communication.\n\n" \
    "This task is about HOW the phrase is written, not what it refers to.\n\n" \
    "MORPHOLOGY GUIDANCE:\n\n" \
    "Strong indicators of \"Expert\":\n" \
    "\t- Greek or Latin diagnostic suffixes (e.g., -itis, -oma, -osis, " \
        "-emia, -pathy, -plegia, -megaly, -penia, -cephaly).\n" \
    "\t- Formal anatomical or pathological terminology primarily used in " \
        "professional contexts.\n" \
    "\t- Standardized clinical disease names.\n\n" \
    "Strong indicators of \"Layperson\":\n" \
    "\t- Common everyday English words (e.g., liver, tumor, cancer, failure, " \
        "rupture, finger, nail, bladder, bowel, leg).\n" \
    "\t- Descriptive constructions like:\n" \
    "\t\t- \"inflammation of ...\"\n" \
    "\t\t- \"tumor of ...\"\n" \
    "\t\t- \"failure\"\n" \
    "\t\t- \"shortening\"\n" \
    "\t\t- \"duplication\"\n" \
    "\t\t- \"abnormality of ...\"\n" \
    "\t\t- \"difference in ...\"\n" \
    "\t\t- Plain descriptive phrasing without specialized suffixes\n\n" \
    "The following words are NOT automatically Expert:\n" \
    "tumor, cancer, failure, rupture, inflammation, duplication, " \
        "abnormality, shortening, asymmetry, hernia.\n" \
    "Formal grammatical structure (e.g., \"Abnormality of X\", \"Rupture " \
        "of Y\") does NOT make a term Expert if it uses plain English " \
        "vocabulary.\n\n" \
    "DECISION PROCESS:\n\n" \
    "Step 1:\n" \
    "Does the synonym contain specialized Greek/Latin medical morphology or " \
        "standardized diagnostic terminology?\n" \
    "\t- If YES, then classify it as \"Expert\".\n" \
    "\t- If NO, then go to Step 2.\n" \
    "Step 2:\n" \
    "Is the synonym written in plain descriptive English that could " \
        "plausibly be used by a patient speaking to a doctor?\n" \
    "- If YES, then classify as Layperson.\n" \
    "- If you are unsure then prefer Layperson unless clear specialist " \
        "terminology is present.\n\n" \
    "CLASS DEFINITIONS\n\n" \
    "\"Layperson\": Uses everyday, non-technical language understandable to " \
        "the general public. May describe medical conditions, but avoids " \
        "specialist morphological terminology.\n" \
    "\"Expert\": Uses technical, clinical, anatomical, Greek/Latin-derived, " \
        "or standardized medical terminology intended primarily for " \
        "healthcare professionals or researchers.\n\n" \
    "Primary HPO concept:\n\n" \
    f"- Label: {quote(label)}\n" \
    f"- Definition: {quote(definition)}\n" \
    f"- Comment: {quote(comment)}\n" \
    f"- Parent concept(s): {applyFormat(parents)}\n" \
    f"- Child concept(s): {applyFormat(children)}\n\n" \
    "Candidate synonym:\n\n" \
    f"- Label: {quote(synonym)}\n\n" \
    "OUTPUT INSTRUCTIONS:\n\n" \
    "Do not include explanations or additional text.\n" \
    "Output exactly one word:\n\n" \
    "\"Layperson\"" \
    "\"Expert\""
"""


