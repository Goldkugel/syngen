import os
import sys
import contextlib
import gc
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

from prompts import *
from utils   import *
from config  import *

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

class Model:

    messageHistories = []

    llm = None
    sampling_params = None
    model = None

    def __init__(self, model : str = model_id):
        self.model = model

        self.llm = LLM(
            model=model, 
            tensor_parallel_size=len(gpu_id.split(",")), 
            max_model_len=max_model_len
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

    def addPrompt(self, role : str = userRole, message : list = []) -> int:
        ret = 0

        if len(message) == 1 and len(self.messageHistories) > 0:
            for index in range(0, len(self.messageHistories)):
                self.messageHistories[index].append({
                    messageRoleElement : role,
                    messageTextElement : message[0]
                })
            ret = len(self.messageHistories)
        else:
            if len(message) == len(self.messageHistories):
                for index in range(0, len(self.messageHistories)):
                    self.messageHistories[index].append({
                        messageRoleElement : role,
                        messageTextElement : message[index]
                    })
                ret = len(self.messageHistories)
            else:
                if len(self.messageHistories) == 0 and len(message) > 0:
                    for index in range(0, len(message)):
                        history = []
                        history.append({
                            messageRoleElement : role,
                            messageTextElement : message[index]
                        })
                        self.messageHistories.append(history)
                    ret = len(self.messageHistories)

        return ret
    
    def generateGemma(self, logging : bool = False) -> None:

        inputs = []
        
        for messageHistory in self.messageHistories:
            prompt = ""
            s = False
            for message in messageHistory:
                if message[messageRoleElement] == systemRole:
                    prompt += f"{startTurn}{userRole}\n" \
                            f"{message[messageTextElement]}\n\n"
                    s = True
                else:
                    if s:
                        prompt +=  f"{message[messageTextElement]}{endTurn}\n"
                        s = False
                    else:
                        prompt += f"{startTurn}{message[messageRoleElement]}\n" \
                            f"{message[messageTextElement]}{endTurn}\n"

            prompt += f"{startTurn}{modelRole}"

            if logging:
                log(prompt)

            inputs.append(prompt)

        # Generate response  
        generatedText = self.llm.generate(inputs, self.sampling_params)
        outputs = []

        for text in generatedText:
            outputs.append(formatGeneratedText(text.outputs[0].text))

        self.addPrompt(role = modelRole, message = outputs)

    def generateLlama(self, logging : bool = False) -> None:

        inputs = []

        for messageHistory in self.messageHistories:
            prompt = f"{beginOfText}"
            
            for message in messageHistory:
                prompt += f"{startHeader}{message[messageRoleElement]}" \
                    f"{endHeader}{message[messageTextElement]}{endOfText}"
                
            prompt += f"{startHeader}{modelRole}{endHeader}"

            if logging:
                log(prompt, cmdline = False)

            inputs.append(prompt)

        # Generate response  
        generatedText = self.llm.generate(inputs, self.sampling_params)
        outputs = []

        for text in generatedText:
            outputs.append(formatGeneratedText(text.outputs[0].text))

        self.addPrompt(role = modelRole, message = outputs)

    def generate(self, logging : bool = False) -> None:
        #if "gemma" in self.model.lower():
        self.generateGemma(logging)
        #else:
        #    self.generateLlama(logging)

    def getMessageHistories(self) -> list[list[object]]:
        return self.messageHistories

    def reset(self) -> None:
        self.messageHistories = []

    def logPrompts(self, file : str = logFilePrompts) -> None:
        for index in range(0, len(self.messageHistories)):
            for index2 in range(0, len(self.messageHistories[index])):
                self.logPrompt(file, index, index2)

    def logPrompt(self, file: str = logFilePrompts, indexHistory : int = 0, indexPrompt : int = -1) -> None:
        message = self.messageHistories[indexHistory][indexPrompt]
        log(f"Role: {message[messageRoleElement]}", False, file)
        log("Message: ", False, file)
        log(message[messageTextElement], False, file)

    def __del__(self) -> None:
        try:
            del self.llm
        except:
            pass
        try:
            destroy_model_parallel()
        except:
            pass
        try:
            destroy_distributed_environment()
        except:
            pass
        try:
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
        except:
            pass
        try:
            gc.collect()
        except:
            pass
        try:
            torch.cuda.empty_cache()
        except:    
           pass
            
def formatGeneratedText(text : str) -> str:
    ret = text.replace(startHeader, "").replace(endHeader, 
        "").replace(endOfText, "").replace(beginOfText, "").replace(modelRole, 
        "").replace(userRole, ""). replace(systemRole, "").strip()
    return ret