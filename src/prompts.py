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
            string = quote((quotationCharacter + ", " + quotationCharacter).join(l))
    return string

def getSystem() -> str:
    return "You are a biomedical ontology expert specializing in clinical " \
        "phenotype terminology and medical vocabulary standardization."

def getTaskPart1(label : str, definition : str) -> str:
    ret = "Let us create a list of synonyms of a concept of a medical " \
        "condition.\n\n" \
    "The medical condition is: {}.\n" \
    "The official definition of the concept is: {}.\n\n" \
    "Are there any synonyms that need to be added to the list? \n\n" \
    "Only include factual synonyms that are exact matches for the medical " \
        "condition. Avoid:\n\n" \
    "- ambiguous synonyms.\n" \
    "- being nonobjective.\n\n" \
    "For each synonym, briefly explain why it qualifies as an exact match."

    return ret.format(quote(label), quote(definition))

def getTaskPart2() -> str:
    return "Review all the synonyms. Classify each synonym " \
        "into one of the following categories:\n\n" \
    "- Clinical terms.\n" \
    "- Lay terms.\n\n" \
    "Explain the decision and identify any missing types or gaps."

def getTaskPart3() -> str:
    return "Now that you have identified missing types and gaps, are there " \
        "other synonyms that would fit into these categories?\n\n" \
    "Once again, only include synonyms that match for the " \
        "medical condition, meaning they refer precisely to the same " \
        "condition and its definition without ambiguity or overlap with " \
        "related but distinct conditions. For each synonym, briefly " \
        "explain why it qualifies as an exact match."

def getTaskPart4(parents : list) -> str:
    ret = "This is the list of parent concept(s) of the medical condition:\n\n" \
    "Parent Concept(s): {}\n\n" \
    "Is there any synonym from the above that is " \
        "overly broad and as a consequence would fit " \
        "better to one of the parent concept(s)?"

    return ret.format(applyFormat(parents))

def getTaskPart5(children : list) -> str:
    ret = "This is the list of the child concept(s) of the medical condition:\n\n" \
    "Child Concept(s): {}\n\n" \
    "Is there any synonym from the above that is overly narrow and, " \
        "consequently, would fit better with one of the child concepts?\n\n" \
    "If no child concept is provided there is nothing to do for you."
    
    return ret.format(applyFormat(children))

def getTaskPart6(examples : list[list[str]] = sourceSynonymExampleResult) -> str:
    ret = "Format the final list of additional synonyms as a comma-separated " \
        "sequence, enclosing each synonym in double quotes.\n\n" \
    "For example:\n\n" \
    "{}\n\n" \
    "Do not include any categories, additional text, or explanation - just " \
        "the formatted list. Use only the information you find in " \
        "the conversation we had."
    
    return ret.format(createExampleString(examples))