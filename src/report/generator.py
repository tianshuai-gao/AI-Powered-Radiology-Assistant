import re


def clean_mri_findings_output(raw_output: str) -> str:
    """
    Extracts the MRI Findings section from raw LLaMA output.
    Looks for text immediately after '**BEGIN OUTPUT**' up to a blank line or end of text.
    """
    pattern = r"(?<=\*\*BEGIN OUTPUT\*\*\n)(.*?)(?=\n\n|$)"
    match = re.search(pattern, raw_output, re.DOTALL)
    return match.group(0).strip() if match else "❌ MRI Findings not found."


def generate_mri_findings(mri_description: str, llama_model) -> str:
    """
    Generate the 'Findings' section of an MRI report using a one-shot LLaMA prompt.
    """
    prompt = f"""
Generate the 'Findings' section of a standard MRI report based on the following MRI imaging description.

**Imaging Description**:
{mri_description}

**BEGIN OUTPUT**
"""
    response = llama_model(prompt, max_length=2048, temperature=0.3)[0]["generated_text"]
    return response


def assess_mri_risk(mri_findings: str, llama_model) -> str:
    """
    Generate the 'Risk Assessment' section of an MRI report based on MRI findings.
    Uses structured prompting with scoring guidelines.
    """
    prompt = f"""
Generate the 'Risk Assessment' section of a standard MRI report based on the following MRI findings.

**MRI Findings**:
{mri_findings}

**BEGIN OUTPUT**
"""
    response = llama_model(prompt, max_length=2048, temperature=0.3)[0]["generated_text"]
    return response


def generate_treatment_plan(mri_findings: str, risk_level: str, llama_model) -> str:
    """
    Generate the 'Treatment Recommendations' section based on MRI findings and risk level.
    Uses one-shot prompting with examples.
    """
    prompt = f"""
Generate the 'Treatment Recommendations' section of a standard MRI diagnostic report.

**MRI Findings**:
{mri_findings}

**Risk Assessment:** {risk_level}

**BEGIN OUTPUT**
"""
    response = llama_model(prompt, max_length=2048, temperature=0.3)[0]["generated_text"]
    return response


def extract_output_section(raw_text: str) -> str:
    """
    Extracts content between '**BEGIN OUTPUT**' and '**END OUTPUT**' markers.
    """
    pattern = r"\*\*BEGIN OUTPUT\*\*(.*?)\*\*END OUTPUT\*\*"
    match = re.search(pattern, raw_text, re.DOTALL)
    return match.group(1).strip() if match else "❗ Unable to locate content."


def generate_mri_diagnostic_report(mri_description: str, llama_model) -> str:
    """
    Generate a full MRI diagnostic report combining Findings, Risk Assessment, and Treatment.
    """
    # Findings
    raw_findings = generate_mri_findings(mri_description, llama_model)
    findings = clean_mri_findings_output(raw_findings)
    # Risk
    raw_risk = assess_mri_risk(findings, llama_model)
    risk = extract_output_section(raw_risk)
    # Treatment
    raw_plan = generate_treatment_plan(findings, risk, llama_model)
    plan = extract_output_section(raw_plan)
    # Combine
    report = (
        "=== MRI Diagnostic Report ===\n\n"
        "**MRI Findings:**\n" + findings + "\n\n"
        "**Risk Assessment:**\n" + risk + "\n\n"
        "**Treatment Recommendations:**\n" + plan + "\n"
        "==============================="
    )
    return report
