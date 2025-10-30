import random
import re
import os
from time import time
from datetime import datetime
from typing import Optional

from langchain.prompts import PromptTemplate
from llms import LLM, LLama, Qwen, Gemma, Mistral, Command
from loguru import logger

SET_NAME = os.getenv("SET_NAME", "blank")

class LLMsType:
    @staticmethod
    def get_llm(llm) -> LLama | Qwen | Gemma | Mistral | Command :
        """Get the LLM type based on the input string."""
        match llm:
            case "command":
                return Command()
            case "mistral":
                return Mistral()
            case "llama3":
                return LLama("llama3")
            case "nemotron":
                return LLama("nemotron")
            case "qwen_qwq":
                return Qwen("qwen_qwq")
            case "qwen-2.5":
                return Qwen("qwen-2.5")
            case "gemma":
                return Gemma()
            case _:
                raise ValueError(f"Unsupported LLM type: {llm}")

    @staticmethod
    def get_llm_token(llm) -> str:
        match llm:
            case "command":
                return "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"
            case "mistral":
                return "[/INST]"
            case "gemma":
                return """<end_of_turn>
            <start_of_turn>model
            """
            case ("llama3" | "nemotron"):
                return """<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            case llm if "qwen" in llm:
                return "<|im_end|>"
            case _:
                raise ValueError(f"Unsupported LLM type: {llm}")


def heavy_filter(response: str) -> str:
    if '[END]' in response:
        response = response.replace("[END]", "[/END]")
    if '[**/END**]' in response:
        response = response.replace("[**/END**]", "[/END]")
    if '[**END**]' in response:
        response = response.replace("[**END**]", "[/END]")
    if '/END' in response:
        response = response.replace("/END", "[/END]")
    if '[[/END]]' in response:
        response = response.replace("[[/END]]", "[/END]")
    if '[]' in response:
        response = response.replace("[]", "[/END]")

    return response


class Agent:
    def __init__(
        self, llm: str, prompt_text: str = "", questionnaire: str = ""
    ) -> None:
        self.llm_type: str = llm
        self.llm: LLM = LLMsType.get_llm(llm)
        self.llm_token: str = LLMsType.get_llm_token(llm)
        self.prompt_text = self.load_prompt(prompt_text)
        self.questionnaire_name = questionnaire.split("/")[-1].split(".")[0]
        self.questionnaire_style = questionnaire.split("/")[-3]
        self.questionnaire = self.load_questionnaire(questionnaire)
        self.prompt_template = PromptTemplate(
            input_variables=["history", "questionnaire"], template=self.prompt_text
        )

    @staticmethod
    def load_prompt(file_name: str) -> str:
        with open(f"{file_name}", "r") as f:
            return f.read()

    @staticmethod
    def load_questionnaire(file_name: str) -> str:
        with open(f"{file_name}", "r") as f:
            return f.read()

    def generate(self, history: list[dict], comment: str = "") -> str:
        prompt: str = self.prompt_template.format(
            questionnaire=self.questionnaire,
            history="\n".join(f"{message['response']}" for message in history),
        )

        if comment != "":
            if "qwen" not in self.llm_type:
                prompt = prompt.split(self.llm_token)[0]
            else:
                parts = prompt.rsplit(self.llm_token, 1)
                if len(parts) > 1:
                    prompt = parts[0] + parts[1]
                else:
                    raise ValueError("No token found in the prompt")
            prompt = f"{prompt}\n{comment}\n{self.llm_token}"
            logger.debug(f"{comment}")
        return self.llm.generate(prompt)

    def post_process(self, response: str, turn) -> str:
        return ""


class Client(Agent):
    def __init__(
        self, llm: str, prompt_text: str = "", questionnaire: str = ""
    ) -> None:
        super().__init__(llm, prompt_text, questionnaire)

    def post_process(self, response: str, turn: int) -> str:
        if "**Client:**" in response:
                response = response.replace("**Client:**", "Client:")

        if "<|END_RESPONSE|>" in response:
            response = response.replace("<|END_RESPONSE|>", "")

        # Remove anything before the client's response
        if self.llm_type == "qwen_qwq":
            match = re.search(r'</think>\s*(.*)', response, re.DOTALL)
            response = match.group(1).strip() if match else response

        match: re.Match[str] | None = re.search(r"(Client: .+)", response)
        if match:
            response = match.group(1).strip()

        if self.llm_type == "qwen":
            # Remove agent <tool_call> token in qwen
            cleaned_response: str = response.replace("<tool_call>", "")
        else:
            cleaned_response = response

        # We only need the client part of the response, we shouldn't let agent generate the THERAPIST response
        redundant_gen_id = cleaned_response.find("Therapist:")
        if redundant_gen_id != -1:  # If the word is found
            cleaned_response = cleaned_response[:redundant_gen_id]

        cleaned_response = heavy_filter(cleaned_response)

        if "mdd" in self.questionnaire_style:
            if turn < 12:
                # Avoid ending the session too early
                cleaned_response = cleaned_response.replace("[/END]", "")
                cleaned_response = cleaned_response.replace("[END]", "")
        else:
            if turn < 7:
                # Avoid ending the session too early
                cleaned_response = cleaned_response.replace("[/END]", "")
                cleaned_response = cleaned_response.replace("[END]", "")

        # Strip spaces and newlines
        cleaned_response = cleaned_response.strip()
        if self.llm_type == "qwen_qwq" or self.llm_type == "glm":
            if self.llm_type == "qwen_qwq" or self.llm_type == "glm":
                pattern = r'Client:.*?\"'
                match = re.search(pattern, cleaned_response, re.DOTALL)

                if match:
                    # Get the matched text but remove the trailing quote
                    cleaned_response = match.group(0)[:-1]  # Remove the last character (quote)
                    cleaned_response = cleaned_response.strip()
                else:
                    # If no match, try a different approach - find the first quote that ends a sentence
                    end_quote_position = cleaned_response.find('."')
                    if end_quote_position != -1:
                        # Return everything up to and including the period but not the quote
                        cleaned_response = cleaned_response[:end_quote_position + 1].strip()
                    else:
                        # Return the original text if no pattern is found
                        cleaned_response = cleaned_response.strip()

            cleaned_response = re.sub(r'\s*\(\d+\s+words\)$', '', cleaned_response)

        # In case the client response is empty
        if cleaned_response == "" or cleaned_response == "Client:":
            cleaned_response = random.choice(
                    [
                        "Client: [No reply]",
                        "Client: [Pause and thinking]",
                        "Client: [Keep silent]",
                        "Client: [Quiet]",
                        "Client: I don't know what to say",
                        "Client: I don't know",
                    ]
                )

        if cleaned_response.startswith("Client:"):
            return cleaned_response
        else:
            return random.choice(
                    [
                        "Client: [No reply]",
                        "Client: [Pause and thinking]",
                        "Client: [Keep silent]",
                        "Client: [Quiet]",
                        "Client: I don't know what to say",
                        "Client: I don't know",
                    ]
                )


class Therapist(Agent):
    def __init__(
        self, llm: str, prompt_text: str = "", questionnaire: str = ""
    ) -> None:
        super().__init__(llm, prompt_text, questionnaire)

    def post_process(self, response: str, turn: int) -> str:
        if "**Therapist:**" in response:
            response = response.replace("**Therapist:**", "Therapist:")

        if "<|END_RESPONSE|>" in response:
            response = response.replace("<|END_RESPONSE|>", "")

        # Remove anything before the therapist response
        if self.llm_type == "qwen_qwq":
            match = re.search(r'</think>\s*(.*)', response, re.DOTALL)
            response = match.group(1).strip() if match else response

        match = re.search(r"(Therapist: .+)", response)
        if match:
            response = match.group(1).strip()

        # We only need the Therapist part of the response, we shouldn't let agent generate the client response
        redundant_gen_id = response.find("Client:")
        if redundant_gen_id != -1:  # If the word is found
            response = response[:redundant_gen_id]

        if self.llm_type == "qwen":
            # Remove agent <tool_call> token in qwen
            response = response.replace("<tool_call>", "")

        response = heavy_filter(response)

        if "mdd" in self.questionnaire_style:
            if turn < 12:
                # Avoid ending the session too early
                response = response.replace("[/END]", "")
        else:
            if turn < 7:
                # Avoid ending the session too early
                response = response.replace("[/END]", "")

        # Strip spaces and newlines
        response = response.strip()

        if self.llm_type == "qwen_qwq" or self.llm_type == "glm":
            pattern = r'Therapist:.*?\"'
            match = re.search(pattern, response, re.DOTALL)

            if match:
                # Get the matched text but remove the trailing quote
                response = match.group(0)[:-1]  # Remove the last character (quote)
                response = response.strip()
            else:
                # If no match, try a different approach - find the first quote that ends a sentence
                end_quote_position = response.find('."')
                if end_quote_position != -1:
                    # Return everything up to and including the period but not the quote
                    response = response[:end_quote_position + 1].strip()
                else:
                    # Return the original text if no pattern is found
                    response = response.strip()

            response = re.sub(r'\s*\(\d+\s+words\)$', '', response)

        return response


class TherapySession:
    def __init__(
        self, client: Client, therapist: Therapist
    ) -> None:
        self.client: Client = client
        self.therapist: Therapist = therapist
        self.history: list[dict] = []

    def _add_to_history(self, role: str, response: str) -> None:
        self.history.append({"role": role, "response": response})

    def out_to_file(self, file_name: str) -> None:
        with open(file_name, "w", encoding="utf-8") as f:
            for message in self.history:
                f.write(f"{message['response']}\n")
        logger.success(f"Output to {file_name}")

    def step(
        self,
        iterative_rewrites,
        inference_object: Agent,
        turn=-1,
        force_ending: str = "",
        repetitive=False,
        repeat_response="",
    ) -> str:
        judge_comment: str = ""
        response: str = ""
        if repeat_response != "" and repetitive:
            judge_comment = f"Your utterance should be different from {repeat_response}"
        if force_ending == "You should end the conversation in this turn":
            judge_comment = force_ending
        elif force_ending != "":
            judge_comment += f"\n{force_ending}"
        for rewrite_turn in range(iterative_rewrites + 1):
            response = inference_object.generate(self.history, comment=judge_comment)
            response = inference_object.post_process(response, turn)
            if rewrite_turn == iterative_rewrites:
                break

            if force_ending == "You should end the conversation in this turn":
                judge_comment = force_ending
            elif force_ending != "" and "[/END]" not in judge_comment:
                judge_comment += f"\n{force_ending}"
            elif force_ending != "" and "[/END]" in response:
                judge_comment = (
                    "You can end the conversation in this turn with '[/END]' token."
                )

            if repetitive:
                judge_comment += f"\nYour utterance should be different from {response}"

        return response

    def run(self, iterative_rewrites: int = 0) -> None:
        start_time = time()
        logger.info("Session started")
        turn = 1
        iterative_rewrites = 1 if random.random() < 0.3 else iterative_rewrites
        therapist_response = self.step(iterative_rewrites, self.therapist, turn)
        self._add_to_history("Therapist", therapist_response)
        logger.info(therapist_response)

        client_response = self.step(iterative_rewrites, self.client, turn)
        self._add_to_history("Client", client_response)
        logger.info(client_response)
        force_ending: str = ""
        while True:
            iterative_rewrites = 1 if random.random() < 0.3 else iterative_rewrites
            therapist_response: str = self.step(
                iterative_rewrites, self.therapist, turn, force_ending=force_ending
            )
            if turn < 15 and "[/END]" in therapist_response:
                therapist_response = therapist_response.replace(" [/END]", "")
            logger.debug(
                "*** Therapist's utterance duplicate: {}",
                {"role": "Therapist", "response": therapist_response} in self.history,
            )
            attempt_count = 0
            max_attempts = 10
            while ({"role": "Therapist",
                    "response": therapist_response,
            } in self.history or not therapist_response.startswith(
                "Therapist:")) and attempt_count < max_attempts:
                logger.warning(
                    f"We are here because the therapist's response is duplicated or not starting with Therapist: , it is: {therapist_response}"
                )
                attempt_count += 1
                therapist_response = self.step(
                    iterative_rewrites,
                    self.therapist,
                    force_ending=force_ending,
                    repetitive=True,
                    repeat_response=therapist_response,
                )

                if attempt_count >= max_attempts:
                    error_msg = (f"Failed to generate unique therapist response after {max_attempts} attempts. "
                                 f"Questionnaire: {self.therapist.questionnaire_name}, LLM: {self.therapist.llm_type}")
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            self._add_to_history("Therapist", therapist_response)
            logger.info(therapist_response)

            client_response: str = self.step(
                iterative_rewrites, self.client, turn, force_ending=force_ending
            )
            turn += 1
            logger.debug(
                "*** Client's utterance duplicate: {}",
                {"role": "Client", "response": client_response} in self.history,
            )

            # Check if the session should end
            if (
                (
                    self.history[-1] == self.history[-3]
                    and self.history[-2] == self.history[-4]
                )
                or {"role": "Client", "response": client_response} in self.history
                or not client_response.startswith("Client:")
            ):
                logger.warning(
                    f"We are here because the client's response is duplicated or not starting with Client: , it is: {client_response}"
                )
                choice = random.choice(
                    [
                        "Client: [Pause and say nothing]",
                        "Client: [Pause and thinking]",
                        "Client: [Sigh]",
                        "Client: [Takes a deep breath]",
                        "Client: I don't know what to say",
                        "Client: I don't know",
                    ]
                )
                self._add_to_history("Client", choice)
                logger.info(choice)
            elif "[/END]" in client_response or "[/END]" in therapist_response:
                logger.success("Conversation takes: {}", turn)
                logger.info(client_response)
                if {
                    "role": "Client",
                    "response": client_response.replace(" [/END]", ""),
                } not in self.history:
                    if "[/END]" in client_response:
                        self._add_to_history("Client", client_response)
                    else:
                        self._add_to_history("Client", client_response + " [/END]")
                else:
                    if "[/END]" not in self.history[-1]["response"]:
                        self.history[-1]["response"] += " [/END]"
                    self._add_to_history("Client", client_response)
                break
            elif turn >= 22:
                logger.warning(
                    "Conversation takes too long, try asking agents to end the conversation"
                )
                logger.info(client_response)
                self._add_to_history("Client", client_response)
                turn_to_end = 30 - turn
                if turn_to_end == 1:
                    force_ending = (
                        f"You should end the conversation in {turn_to_end} turn."
                    )
                elif turn_to_end == 0:
                    force_ending = "You should end the conversation in this turn with '[/END]' token."
                else:
                    force_ending = (
                        f"You should end the conversation in {turn_to_end} turns."
                    )
            else:
                logger.info(client_response)
                self._add_to_history("Client", client_response)
            if turn >= 33:
                logger.error(
                    "Conversation takes too long, force ending the conversation"
                )
                break
            logger.info("Turn: {}", turn)

        logger.info("Session ended, out to file")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

        if not os.path.isdir(f"syn_gen_{SET_NAME}"):
            os.makedirs(f"syn_gen_{SET_NAME}")
            print(f"Created directory: syn_gen_{SET_NAME}")
        else:
            print(f"Directory already exists: syn_gen_{SET_NAME}")

        file_path = (
            f"syn_gen_{SET_NAME}/"
            f"{current_time}_data_{self.therapist.questionnaire_name}_"
            f"therapist_{self.therapist.llm_type}_client_{self.client.llm_type}.txt"
        )

        self.out_to_file(file_path)

        # Calculate elapsed time
        end_time = time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        logger.info(
            "Total time elapsed: {:02d}:{:02d}:{:02d}",
            hours, minutes, seconds
        )
        logger.success("Session ended")
