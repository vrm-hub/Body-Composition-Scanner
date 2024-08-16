from openai import AsyncOpenAI

# add API Key in environment variable: OPENAI_API_KEY
client = AsyncOpenAI()


async def generate_health_report(metrics):
    prompt = (
        f"Generate a detailed health report based on the following metrics:\n"
        f"Neck Circumference: {metrics['Neck']} cm\n"
        f"Waist Circumference: {metrics['Waist']} cm\n"
        f"Hip Circumference: {metrics['Hip']} cm\n"
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
        "Please include a summary of the individual's health status, potential health risks, and any recommended actions or lifestyle changes."
    )

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a health analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.5
    )

    return response.choices[0].message.content.strip()
