import os
import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

useOutputAsInput = True
generateTimes = 50

inputFile = "~/syngen/data/input/hpo.data.pkl"
outputFile = "~/syngen/data/output/hpo.data.enriched.pkl"

#model_id = "meta-llama/Llama-3.2-1B"
#model_id = "meta-llama/Llama-3.2-3B"
#model_id = "meta-llama/Llama-3.1-8B"
#model_id = "meta-llama/Llama-3.1-70B"
#model_id = "meta-llama/Llama-3.2-1B-Instruct"
#model_id = "meta-llama/Llama-3.2-3B-Instruct"
#model_id = "meta-llama/Llama-3.1-8B-Instruct"
#model_id = "meta-llama/Llama-3.3-70B-Instruct"
model_id = "google/medgemma-27b-text-it"
#model_id = "lingshu-medical-mllm/Lingshu-32B"

#gpu_id = "1,2,3,4"
gpu_id = "4,5,6,7"
#gpu_id = "6,7"

logFile = "~/syngen/data/hpot.log"
logFilePrompts = "~/syngen/data/prompts.log"

# Float that controls the cumulative probability of the top tokens to consider.
# Must be in (0, 1]. Set to 1 to consider all tokens.
top_p=0.9

max_model_len=32768
max_num_batched_tokens=2*max_model_len

# Float that controls the randomness of the sampling. Lower values make the 
# model more deterministic, while higher values make the model more random. 
# Zero means greedy sampling.
temperature = 0.9

quotationCharacter = "\""

systemRole = "system"
userRole = "user"
modelRole = "assistant"

# For Gemma
startTurnID = "start_of_turn"
endTurnID = "end_of_turn"

# For Llama
startHeaderID = "start_header_id"
endHeaderID = "end_header_id"
endOfTextID = "eot_id"
beginOfTextID = "begin_of_text"
endOfTextID2 = "end_of_text"

startTag = "<"
endTag = ">"
bar = "|"

# For Gemma
startTurn = f"{startTag}{startTurnID}{endTag}"
endTurn = f"{startTag}{endTurnID}{endTag}"

# For Llama
startHeader = f"{startTag}{bar}{startHeaderID}{bar}{endTag}"
endHeader = f"{startTag}{bar}{endHeaderID}{bar}{endTag}"
endOfText = f"{startTag}{bar}{endOfTextID}{bar}{endTag}"
beginOfText = f"{startTag}{bar}{beginOfTextID}{bar}{endTag}"
endOfText2 = f"{startTag}{bar}{endOfTextID2}{bar}{endTag}"

messageRoleElement = "role"
messageTextElement = "message"

# Random seed to use for the generation
seed = 2898231092

# Maximum number of tokens to generate per output sequence.
max_tokens = 2048

headerChar = "="
headerLen = 80
headerSeparator = headerChar * headerLen

progressBarColor = "cyan"
progressBarTextLength = 20

universalLanguage = "universal"

languageColumn  = "language"
classColumn     = "class"
contentColumn   = "content"
hpoidColumn     = "hpoID"
elementIDColumn = "id"
sourceElemement = "source"
systemColumn = "system"
countColumn = "count"

labelClass                      = "label"
definitionClass                 = "definition"
commentClass                    = "comment"
exactSynonymClass               = "exactSynonym"
enrichedSourceExactSynonymClass = "sourceArtificialSynonym"
rephrasingSourceSynonymClass    = "sourceRephrasing"
referenceClass                  = "reference"
relatedSynonymClass             = "relatedSynonym"
childrenClass                   = "child"
alternativeSpellingClass        = "alternativeSpelling"
enrichedTargetExactSynonymClass = "targetArtificialSynonym"
rephrasingTargetSynonymClass    = "targetRephrasing"
translationClassPrefix          = "translation"
enrichedSourceDefinitionClass   = "sourceArtificialDefinition"

# Basic Data Directory.
dataDir = "~/syngen/data"

sourceSynonymExampleResult = [
    [
        "Apple", 
        "Banana", 
        "Pineapple",
        "Peach"
    ], [
        "Pepper",
        "Eggplant",
        "Cucumber",
        "Tomato"
    ], [
        "Blueberry",
        "Strawberry",
        "Blackberry",
        "Raspberry"
    ]
]