from openai import AsyncOpenAI

# add API Key in environment variable: OPENAI_API_KEY
client = AsyncOpenAI()

async def generate_health_report(metrics):
    prompt = (
        f"Generate a detailed health report based on the following metrics:\n"
        f"Height: {metrics['Height']} cm\n"
        f"Weight: {metrics['Weight']} kg\n"
        f"Body Fat Percentage: {metrics['BFP']}%\n"
        f"Essential Fat: {metrics['Essential_Fat']} kg\n"
        f"Beneficial Fat: {metrics['Beneficial_Fat']} kg\n"
        f"Unbeneficial Fat: {metrics['Unbeneficial_Fat']} kg\n"
        f"Lean Mass: {metrics['Lean_Mass']} kg\n"
        f"Lean Mass Index (LMI): {metrics['LMI']} kg/m^2\n"
        f"Fat Mass Index (FMI): {metrics['FMI']} kg/m^2\n"
        f"Resting Metabolic Rate (RMR): {metrics['RMR']} kcal/day\n\n"
        "Please provide the report in a structured and readable format, including:\n"
        "1. A summary of the individual's current health status.\n"
        "2. A comparison of each metric with normal levels, indicating how high or low they are.\n"
        "3. An estimate of how long it will ideally take to return to a normal state for each metric that is out of the normal range.\n"
        "4. Potential health risks associated with the current metrics.\n"
        "5. Recommended actions or lifestyle changes to improve health and reach ideal levels.\n"
    )

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a fitness and health expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.5
    )

    return response.choices[0].message.content.strip()

