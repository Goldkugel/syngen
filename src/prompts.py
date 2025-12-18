import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from config import *

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

def getSystem() -> str:
    return "You are a biomedical ontology expert specializing in clinical " \
        "phenotype terminology and medical vocabulary standardization."

def getTaskPart1(label : str, definition : str) -> str:
    return "Let us create a list of synonyms of a medical " \
        "condition suitable for the Human Phenotype Ontology (HPO).\n\n" \
    "The medical condition is: {}.\n" \
    "The official definition of the concept is: {}.\n\n" \
    "Are there synonyms that you would recommend as synonym for the " \
        "medical condition? \n\n" \
    "Please, only include factual synonyms that are exact matches for the medical " \
        "condition and different spelling variants e.g. 'behavior' and 'behaviour' . Avoid:\n\n" \
    "- ambiguous synonyms.\n" \
    "- using IDs of ontologies or terminologies for reference.\n" \
    "- using brackets.\n" \
    "- being nonobjective.\n\n" \
    "For each synonym, briefly explain why it qualifies as an exact match.". \
        format(quote(label), quote(definition))

def getTaskPart2() -> str:
    return "Review all the synonyms. Classify each synonym " \
        "into one of the following categories:\n\n" \
    "- Clinical terms.\n" \
    "- Ontological terms.\n" \
    "- Lay terms.\n\n" \
    "Explain the decision and identify any missing types or gaps."

def getTaskPart3() -> str:
    return "Review all the synonyms. Please remove any synonym you " \
        "classified as \'Lay term\', contain brackets or IDs of ontologies " \
        "/ terminologies since they are not ment to be listed " \
        "in an ontology. \n\n" \
    "Explain the decision and identify any missing types or gaps."

def getTaskPart4() -> str:
    return "Now that you have identified missing types and gaps, are there " \
        "other synonyms that would fit into these categories?\n\n" \
    "Once again, only include synonyms that match for the " \
        "medical condition, meaning they refer precisely to the same " \
        "condition and its definition without ambiguity or overlap with " \
        "related but distinct conditions. For each synonym, briefly " \
        "explain why it qualifies as an exact match."

def getTaskPart5(parents : list) -> str:
    return "This is the list of parent concept(s) of the medical condition:\n\n" \
    "Parent Concept(s): {}\n\n" \
    "Is there any synonym from the above that is " \
        "overly broad and as a consequence would fit " \
        "better to one of the parent concept(s)?".format(applyFormat(parents))

def getTaskPart6(children : list) -> str:
    return "This is the list of the child concept(s) of the medical condition:\n\n" \
    "Child Concept(s): {}\n\n" \
    "Is there any synonym from the above that is overly narrow and, " \
        "consequently, would fit better with one of the child concepts?\n\n" \
    "If no child concept is provided there is nothing to do for you.". \
        format(applyFormat(children))

def getTaskPart7(examples : list[list[str]] = sourceSynonymExampleResult) -> str:
    return "Format the final list of additional synonyms as a comma-separated " \
        "sequence, enclosing each synonym in double quotes.\n\n" \
    "For example:\n\n" \
    "{}\n\n" \
    "Do not include any categories, additional text, or explanation - just " \
        "the formatted list. Use only the information you find in " \
        "the conversation we had.".format(createExampleString(examples))

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


def getAlternativeEasyPrompt1(
        label : str, 
        definition : str, 
        parents : list = [], 
        children : list = []
) -> str:
    return \
    "You are helping to generate EXACT synonyms for an HPO term.\n\n" \
    f"[HPO LABEL]: {quote(label)}\n" \
    f"[HPO DEFINITION]: {quote(definition)}\n" \
    f"[PARENT TERMS]: {applyFormat(parents)}\n" \
    f"[CHILD TERMS]: {applyFormat(children)}\n\n" \
    "Your task:\n" \
    "Generate only exact synonyms for the HPO label. An exact synonym " \
        "must:\n\n" \
    "1. Have the same meaning as the label.\n" \
    "2. Match the definition exactly.\n" \
    "3. Not be broader or narrower.\n" \
    "4. Not add ideas about cause, severity, diagnosis, or timing.\n" \
    "5. Not overlap with any child or parent term.\n" \
    "6. Use standard biomedical wording.\n\n" \
    "Do NOT generate related terms, broader terms, child-term or parent-term " \
        "wording, test names, or causal descriptions.\n\n" \
    "Output only synonyms that could replace the HPO label with no change " \
        "in meaning."

def getAlternativeEasyPrompt2() -> str: 
    return \
    "You now have generated a list of candidate synonyms.\n" \
    "Remove any candidate that does not fully match the meaning of " \
        "the HPO label.\n\n" \
    "Remove a candidate if it:\n\n" \
    "1. Changes the meaning of the label in any way.\n" \
    "2. Adds or removes ideas (cause, severity, diagnosis, timing, " \
        "anatomy, etc.).\n" \
    "3. Is broader or narrower than the label.\n" \
    "4. Sounds like or overlaps with any child or parent term.\n" \
    "5. Is not standard biomedical wording.\n" \
    "6. Is unclear, unusual, or ambiguous.\n\n" \
    "Keep only the candidates that:\n\n" \
    "• have exactly the same meaning as the original label, and\n" \
    "• could replace the label with no change in interpretation.\n\n" \
    "Output only the remaining candidate synonyms."

def getAlternativeEasyPrompt3() -> str: 
    return \
    "Create rephrased versions of the generated synonyms.\n" \
    "These rephrasings must:\n\n" \
    "1. Keep exactly the same meaning as the original label.\n" \
    "2. Match the definition fully.\n" \
    "3. Not add or remove any idea (cause, severity, timing, diagnosis, " \
        "anatomy).\n" \
    "4. Not overlap with any child or parent term.\n" \
    "5. Use normal biomedical phrasing.\n\n" \
    "Allowed types of rephrasing:\n\n" \
    "• switch word order (e.g., “Pubertal delay” → “Delayed puberty”)\n" \
    "• use a standard noun phrase vs. adjective form\n" \
    "• consider spelling variations, e.g. 'tumor' vs. 'tumour'\n" \
    "• make simple, common biomedical wording variants\n\n" \
    "Do NOT create new meanings or specialized/rare expressions.\n" \
    "Add the repharsed terms to the list of candidate synonyms."

def getAlternativeEasyPrompt4() -> str: 
    return \
    "Check each of your generated synonyms and rephrased terms " \
        "carefully.\n\n" \
    "Remove any candidate that:\n\n" \
    "1. Does NOT have exactly the same meaning as the HPO label.\n" \
    "2. Does NOT fully match the definition.\n" \
    "3. Adds or removes any idea (cause, severity, timing, diagnosis, " \
        "anatomy, etc.).\n" \
    "4. Overlaps with or implies any child or parent term.\n" \
    "5. Uses non-standard or unclear biomedical wording.\n\n" \
    "Keep only candidates that are:\n\n" \
    "• fully interchangeable with the original label, and\n" \
    "• valid in any phenotype-annotation context.\n\n" \
    "Output only the remaining synonyms."

def getAlternativeEasyPrompt5() -> str:
    return \
    "Now place the remaining candidate synonyms into the following " \
        "simple structure.\n\n" \
    "Follow these rules:\n\n" \
    "• Do NOT explain how you chose the synonyms.\n" \
    "• Do NOT include reasoning or thought process.\n" \
    "• Use ONLY the fields shown below.\n" \
    "• Do NOT add extra text outside the JSON object.\n\n" \
    "Output exactly this JSON format:\n\n" \
    "{\n" \
    "\t'exact_synonyms': [\n" \
    "\t\t'<<list the remaining exact synonyms>>'" \
    "\t]\n" \
    "}\n\n" \
    "EXAMPLE OUTPUT (for demonstration only):\n\n" \
    "{\n" \
    "\t'exact_synonyms': [\n" \
    "\t\t'Delayed puberty',\n" \
    "\t\t'Pubertal delay',\n" \
    "\t\t'Late onset of puberty'\n" \
    "\t]\n" \
    "}"

def getAlternativeComplexPrompt1(
        label : str, 
        definition : str, 
        parents : list = [], 
        children : list = []
) -> str:
    return "You are an ontology-focused biomedical language model tasked " \
        "with generating EXACT synonyms for a Human Phenotype Ontology " \
        "(HPO) term.\n\n" \
    "Below is the information about the target HPO term.\n\n" \
    f"[HPO LABEL]: {quote(label)}\n" \
    f"[HPO DEFINITION]: {quote(definition)}\n" \
    f"[PARENT TERMS]: {applyFormat(parents)}\n" \
    f"[CHILD TERMS]: {applyFormat(children)}\n\n" \
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
    "\t'exact_synonyms': [\n" \
    "\t\t'<<list the remaining exact synonyms>>'" \
    "\t]\n"\
    "}\n\n" \
    "EXAMPLE OUTPUT (for demonstration only):\n\n" \
    "{\n" \
    "\t'exact_synonyms': [\n" \
    "\t\t'Delayed puberty',\n" \
    "\t\t'Pubertal delay',\n" \
    "\t\t'Late onset of puberty'\n" \
    "\t]\n" \
    "}\n\n" \
    "Produce exactly one JSON object as specified above."