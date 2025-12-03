import os
import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

useOutputAsInput = True
generateTimes = 50

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



# Float that controls the cumulative probability of the top tokens to consider.
# Must be in (0, 1]. Set to 1 to consider all tokens.
top_p=0.9

max_model_len = 32768
max_num_batched_tokens = 2 * max_model_len

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
dataDir = "../data"

logFile = f"{dataDir}/hpot.log"
logFilePrompts = f"{dataDir}/prompts.log"

inputFile = f"{dataDir}/input/hpo.data.pkl"
outputFileGenerated = f"{dataDir}/output/generated.csv"
outputFileGold = f"{dataDir}/output/gold.csv"
outputFileGeneratedEmbeddings = f"{dataDir}/output/generatedembeddings.csv"
outputFileGoldEmbeddings = f"{dataDir}/output/goldembeddings.csv"

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

testIDs = list(set([
    'HP:0001756', 'HP:0003189', 'HP:0000708', 'HP:0008069', 'HP:0009778',
    'HP:0008331', 'HP:0007165', 'HP:0002020', 'HP:0010759', 'HP:0002659']))
"""
    'HP:0009891', 'HP:0007018', 'HP:0032514', 'HP:0000434', 'HP:0000692', 
    'HP:0025675', 'HP:0001511', 'HP:0000286', 'HP:0000448', 'HP:0000691', 
    'HP:0001092', 'HP:0001315', 'HP:0000683', 'HP:0000252', 'HP:0000239', 
    'HP:0002460', 'HP:0008180', 'HP:0025190', 'HP:0000722', 'HP:0009588', 
    'HP:0008368', 'HP:0001631', 'HP:0000689', 'HP:6000990', 'HP:0000260', 
    'HP:0006334', 'HP:0002069', 'HP:0000363', 'HP:0009650', 'HP:0011468', 
    'HP:0012407', 'HP:0000474', 'HP:0009611', 'HP:0003270', 'HP:0000637', 
    'HP:0006347', 'HP:0008066', 'HP:0008070', 'HP:0000733', 'HP:0002745', 
    'HP:0001357', 'HP:0008163', 'HP:0002395', 'HP:0009381', 'HP:0000496', 
    'HP:0000347', 'HP:0002213', 'HP:0005484', 'HP:0001274', 'HP:0005772', 
    'HP:0010808', 'HP:0006897', 'HP:0033010', 'HP:0002123', 'HP:0011160', 
    'HP:0000766', 'HP:0007906', 'HP:0007479', 'HP:0000954', 'HP:0010628', 
    'HP:0000455', 'HP:0011072', 'HP:0032662', 'HP:0005278', 'HP:0008127', 
    'HP:0001047', 'HP:0000198', 'HP:0100832', 'HP:0000421', 'HP:0006487', 
    'HP:0002788', 'HP:0002719', 'HP:0000550', 'HP:0000653', 'HP:0000494', 
    'HP:0001999', 'HP:0000158', 'HP:0031354', 'HP:5200291', 'HP:0012210', 
    'HP:0002683', 'HP:0030447', 'HP:0007628', 'HP:0034348', 'HP:0001520', 
    'HP:0000215', 'HP:0000175', 'HP:0010674', 'HP:0000349', 'HP:0100490', 
    'HP:0010650', 'HP:0000848', 'HP:0001566', 'HP:0012809', 'HP:0009931', 
    'HP:0011332', 'HP:0004209', 'HP:0005285', 'HP:0033349', 'HP:0002376', 
    'HP:0006682', 'HP:0007700', 'HP:0032152', 'HP:0001344', 'HP:0011451', 
    'HP:0009836', 'HP:0006986', 'HP:0001964', 'HP:0007126', 'HP:0011169', 
    'HP:0002495', 'HP:0000973', 'HP:0002119', 'HP:0002089', 'HP:0000275', 
    'HP:0002121', 'HP:0000272', 'HP:0011120', 'HP:0006323', 'HP:0200039',
    'HP:0001841', 'HP:0000444', 'HP:0000528', 'HP:0003812', 'HP:0009940', 
    'HP:0000283', 'HP:0000736', 'HP:0010757', 'HP:0001141', 'HP:0006329', 
    'HP:0008873', 'HP:0002936', 'HP:0006482', 'HP:0006262', 'HP:0002286', 
    'HP:0010701', 'HP:0004315', 'HP:0003565', 'HP:0011832', 'HP:0005261', 
    'HP:0002166', 'HP:0000179', 'HP:0001510', 'HP:0011831', 'HP:0020220', 
    'HP:0000601', 'HP:0000551', 'HP:0000519', 'HP:0009237', 'HP:0004691', 
    'HP:0001382', 'HP:0010885', 'HP:0010107', 'HP:0011675', 'HP:0000418', 
    'HP:0003557', 'HP:0002349', 'HP:0010049', 'HP:0001106', 'HP:0011823', 
    'HP:0000337', 'HP:0000882', 'HP:0031843', 'HP:0000745', 'HP:0012799', 
    'HP:0012810', 'HP:0000384', 'HP:0010289', 'HP:0000270', 'HP:0001622', 
    'HP:0003473', 'HP:0010055', 'HP:0008527', 'HP:0006721', 'HP:0410334', 
    'HP:0002925', 'HP:0002061', 'HP:0100716', 'HP:0000256', 'HP:0000670', 
    'HP:0001988', 'HP:0005914', 'HP:0001308', 'HP:0010669', 'HP:0000520', 
    'HP:0011432', 'HP:0009612', 'HP:0003022', 'HP:0008672', 'HP:0000992', 
    'HP:0005978', 'HP:0034984', 'HP:0009102', 'HP:0006297', 'HP:0000377', 
    'HP:0000430', 'HP:0010702', 'HP:0002910', 'HP:0002283', 'HP:0000680', 
    'HP:0001249', 'HP:0010813', 'HP:0005280', 'HP:0003710', 'HP:0002384', 
    'HP:0008905', 'HP:0000292', 'HP:0430028', 'HP:0003301', 'HP:0011073', 
    'HP:0006308', 'HP:0002705', 'HP:0100543', 'HP:0006288', 'HP:0011159', 
    'HP:0010741', 'HP:0009085', 'HP:0011829', 'HP:0003083', 'HP:0400000', 
    'HP:0000654', 'HP:0000975', 'HP:0012471', 'HP:0001840', 'HP:0010751', 
    'HP:0000995', 'HP:0006349', 'HP:0000463', 'HP:0032795', 'HP:0002216', 
    'HP:0002091', 'HP:0006321', 'HP:0004488', 'HP:0003781', 'HP:0004227', 
    'HP:0009058', 'HP:0009939', 'HP:0001831', 'HP:0003141', 'HP:0003236', 
    'HP:0000698', 'HP:0000278', 'HP:0006313', 'HP:0100851', 'HP:0100540', 
    'HP:0007334', 'HP:0001162', 'HP:0003042', 'HP:0001263', 'HP:0000171', 
    'HP:0030215', 'HP:0008151', 'HP:0033658', 'HP:0001000', 'HP:0003281', 
    'HP:0000750', 'HP:0001900', 'HP:0001288', 'HP:0002711', 'HP:0000544', 
    'HP:0002500', 'HP:0000940', 'HP:0000188', 'HP:0005819', 'HP:0011173', 
    'HP:0002718', 'HP:0009933', 'HP:0100804', 'HP:0033757', 'HP:0100400', 
    'HP:0003202', 'HP:0004467', 'HP:0003025', 'HP:0002684', 'HP:0000529', 
    'HP:0000303', 'HP:0000402', 'HP:0008419', 'HP:0010755', 'HP:0003180', 
    'HP:0033009', 'HP:0000408', 'HP:0008518', 'HP:0000972', 'HP:0004370', 
    'HP:0009882', 'HP:0005989', 'HP:0010743', 'HP:0001290', 'HP:0001558', 
    'HP:0001188', 'HP:0003100', 'HP:0003387', 'HP:0011800', 'HP:0004808', 
    'HP:0001627', 'HP:0004220', 'HP:0010819', 'HP:0003693', 'HP:0000193', 
    'HP:0003319', 'HP:0002953', 'HP:0002757', 'HP:0005465', 'HP:0011222', 
    'HP:0007359', 'HP:0001133', 'HP:0001377', 'HP:0000358', 'HP:0400001', 
    'HP:0030028', 'HP:0003375', 'HP:0010313', 'HP:0008921', 'HP:0005736', 
    'HP:0430029', 'HP:0008209', 'HP:0001270', 'HP:0000403', 'HP:0000426', 
    'HP:0003324', 'HP:0003700', 'HP:0000574', 'HP:0009177', 'HP:0002010', 
    'HP:0001363', 'HP:0009930', 'HP:0000191', 'HP:0005323', 'HP:0005792', 
    'HP:0000276', 'HP:0002209', 'HP:0000221', 'HP:0002197', 'HP:0001338', 
    'HP:0009803', 'HP:0004319', 'HP:0000356', 'HP:0000431', 'HP:0005152', 
    'HP:0012720', 'HP:0006380', 'HP:0011166', 'HP:0007814', 'HP:0005790', 
    'HP:0002720', 'HP:0009019', 'HP:0003015', 'HP:0000119', 'HP:0003261', 
    'HP:0002984', 'HP:0030318', 'HP:0002398', 'HP:0009835', 'HP:0010034', 
    'HP:0000457', 'HP:0003774', 'HP:0009920', 'HP:0000445', 'HP:0010537', 
    'HP:0005274', 'HP:0006498', 'HP:0001762', 'HP:0012292', 'HP:0002694', 
    'HP:0000274', 'HP:0011167', 'HP:0030319', 'HP:0000327', 'HP:0000319', 
    'HP:0100723', 'HP:0000490', 'HP:0040217', 'HP:0003762', 'HP:0034353', 
    'HP:0011069', 'HP:0006532', 'HP:0010972', 'HP:0000324', 'HP:0002094', 
    'HP:0000684', 'HP:0000582', 'HP:0001476', 'HP:0003200', 'HP:0001195', 
    'HP:0006335', 'HP:0000010', 'HP:0030393', 'HP:0002282', 'HP:0006315', 
    'HP:0009601', 'HP:0006336', 'HP:0002681', 'HP:0008551', 'HP:0009244', 
    'HP:0011219', 'HP:0012811', 'HP:0000341', 'HP:0000762', 'HP:0002194', 
    'HP:0002682', 'HP:0003687', 'HP:0410246', 'HP:0000687', 'HP:0000413', 
    'HP:0000218', 'HP:0000316', 'HP:0001254', 'HP:0009746', 'HP:0002922', 
    'HP:0004313', 'HP:0001635', 'HP:0002553', 'HP:0007378', 'HP:0002750', 
    'HP:0001852', 'HP:0007968', 'HP:0000592', 'HP:0009843', 'HP:0005272',
    'HP:0004331', 'HP:0011153', 'HP:0003155', 'HP:0000437'
]))"""