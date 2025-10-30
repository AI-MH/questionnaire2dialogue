# Roleplaying with Structure: Synthetic Therapist-Client Conversation Generation from Questionnaires

## How to Generate

Make sure to has a running VLLM instance
```
PORT=9000
python -m vllm.entrypoints.openai.api_server --model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" --tensor-parallel-size 2 --pipeline-parallel-size 2 --seed 666  --port $PORT
```
You can modify the `PORT` to `openai_api_base` in `llms.py`

### To run the code

```
cd src/
python main.py \
    -m deepseek \
    -pc prompts/deepseek_demographic_client_prompt.txt \
    -pt prompts/deepseek_therapist.txt \
    -q patients/marburg_patient_0.md
```

### Arguments:

```
    --seed, type=int, default=666
    Initial seed

    -m, --model, type=str
    Model for therapist and client agent ('qwen', 'llama', etc.)

    -pc, --prompt_client, type=str
    Prompt use for client agent (see prompts/*client*.txt )

    -pt, --prompt_therapist, type=str
    Prompt use for therapist agent (see prompts/*therapist*.txt folder)

    -ir, --iterative_rewrites, type=int, default=0
    Number of rewrite we want therapist/client to rewrite according to judge comment

    -q, --questionnaire, type=str
    Patient questionnaire (see patients/ folder)

```
