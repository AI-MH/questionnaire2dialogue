# Ethics Statement

This document describes the ethical considerations behind **SQPsych** (the generation pipeline), **SQPsychConv** (the synthetic dialogue corpus), and **SQPsychLLM** (the fine-tuned models) released in this repository. It accompanies the paper *Roleplaying with Structure: Synthetic Therapist-Client Conversation Generation from Questionnaires*.

## 1. Ethics approval

This study was reviewed and approved by the ethics committee of the **Technical University of Darmstadt**.

## 2. Data provenance and participant protection

- **Source:** The structured clinical inputs used to condition generation (BDI and HAM-D scores and demographic metadata) derive from the cohort reported in [**Kircher et al. (2019)**](https://pubmed.ncbi.nlm.nih.gov/30267149/). That cohort was collected under its own ethics approval and participant informed-consent
  procedures; only **de-identified, pre-anonymized structured variables** were made available to this project.
- **No personally identifiable information:** The conditioning data contains no direct identifiers and no free-text patient narratives. No raw clinical records, audio, or identifiable material were used or released.
- **No re-identification:** The released artifacts do not permit reconstruction of any individual participant's data.

## 3. Privacy-preserving design

Data-governance and privacy regulations prohibit transmitting clinical questionnaire data (FOR2107) to third-party or proprietary services. To respect this:

- Clinical inputs are **never** sent to external or proprietary APIs.
- All generation and fine-tuning run on **locally hosted with vLLM, open-weight models**.

## 4. What is released

- **SQPsychConv:** *synthetic* therapist-client conversations produced by the pipeline. These are generated text; they contain no real patient utterances
  and no personally identifiable information.
- **SQPsychLLM:** models fine-tuned from open-weight base models on the synthetic corpus. They are derivative works of their respective base models.
- **Code:** the generation pipeline.

## 5. Intended use

These resources are released **for research and educational purposes only**, for example: studying privacy-preserving synthetic data generation, training and evaluating mental-health dialogue models, and clinician/student training simulations in controlled, supervised environments.

## 6. Out-of-scope and prohibited uses

These resources are **not** a medical device, a diagnostic tool, or a substitute for professional mental-health care. In particular, they must **not** be:

- deployed to interact with real patients or any person in distress without rigorous additional validation, qualified clinical oversight, and appropriate regulatory approval;
- used as a crisis, emergency, or safety-critical support system;
- presented as, or used to impersonate, advice from a real licensed clinician;
- relied upon for any clinical decision.

## 7. Risks and limitations

- **Hallucination and unsafe output:** Like all LLMs, these models can produce incorrect, fabricated, or clinically inappropriate content, including advice that is not evidence-based.
- **Limited clinical scope:** The conditioning data covers **major depressive disorder**; other conditions, comorbidities, severities, and acute risk presentations are out of scope and not represented.
- **Limited population coverage:** The source cohort's demographics, language, and cultural context constrain generalization; the resources should not be assumed valid for other populations.
- **Inherited bias:** Outputs reflect biases present in the open-weight base models and in the source cohort.
- **Synthetic vs real gap:** Generated dialogues may not capture the full complexity, ambiguity, or risk dynamics of real clinical interactions.

## 8. Responsible-use guidance for downstream users

If you build on these resources, you are responsible for using them safely. We ask that you:

- keep a **qualified human professional in the loop** for any applied use;
- obtain your **own ethics approval** and validate on your own population before any study involving human participants;
- implement **safety guardrails and crisis-resource handling** in any interactive system, and do not rely on these models for safety-critical
  behavior;
- clearly disclose to any end user that they are interacting with an AI system, not a clinician.

## 9. Licensing

Use of this repository is governed by the `LICENSE` file (Apache-2.0). The released datasets and models carry also Apache-2.0 license. Note that the licenses and acceptable-use terms of the **base models** used for generation and fine-tuning also apply to derivative artifacts; downstream users must comply with those upstream terms as well.

## 10. Reporting concerns and contact

For questions about these ethical considerations, concerns about the data or models, or takedown requests, please open an issue in this GitHub repo. We will respond to reports and act on legitimate concerns.
