import argparse
import random

import numpy as np
import torch
from agents import Client, Therapist, TherapySession


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(
    model,
    prompt_client,
    prompt_therapist,
    iterative_rewrites,
    questionnaire,
):
    therapist = Therapist(
        llm=model, prompt_text=prompt_therapist, questionnaire=questionnaire
    )
    client = Client(llm=model, prompt_text=prompt_client, questionnaire=questionnaire)
    session = TherapySession(client, therapist)
    session.run(iterative_rewrites)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-pc", "--prompt_client", type=str)
    parser.add_argument("-pt", "--prompt_therapist", type=str)
    parser.add_argument("-ir", "--iterative_rewrites", type=int, default=0)
    parser.add_argument("-q", "--questionnaire", type=str)
    args = parser.parse_args()
    seed_everything(args.seed)
    main(
        args.model,
        args.prompt_client,
        args.prompt_therapist,
        args.iterative_rewrites,
        args.questionnaire,
    )
