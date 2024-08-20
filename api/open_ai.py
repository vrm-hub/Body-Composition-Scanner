from openai import AsyncOpenAI

# Add API Key in environment variable: OPENAI_API_KEY
client = AsyncOpenAI()

async def generate_health_report(metrics, age, gender):
    prompt = (
        f"Hey! Your Body Scan Report is ready to go. Are you?\n\n"f"Sex: {gender}\n"f"Age: {age}\n"f"Height: {metrics['Height']} cm\n"f"Weight: {metrics['Weight']} kg\n"f"Activity Level: Advanced\n\n""First up, let’s look at your body composition:\n"f"Essential Fat: {metrics['Essential_Fat']} kg ({metrics['Essential_Fat']/metrics['Weight']*100:.2f}%)\n"f"Beneficial Fat: {metrics['Beneficial_Fat']} kg ({metrics['Beneficial_Fat']/metrics['Weight']*100:.2f}%)\n"f"Unbeneficial Fat: {metrics['Unbeneficial_Fat']} kg ({metrics['Unbeneficial_Fat']/metrics['Weight']*100:.2f}%)\n"f"Lean Mass: {metrics['Lean_Mass']} kg ({metrics['Lean_Mass']/metrics['Weight']*100:.2f}%)\n\n""Numbers only tell part of the story. Let’s see what they mean:\n""Essential Fat is critical for your body’s functions such as hormone regulation and immune support.\n""Beneficial Fat helps maintain energy levels and supports metabolic functions.\n""Unbeneficial Fat can lead to health issues like cardiovascular disease.\n""Lean Mass contributes to a higher metabolic rate and stronger bones.\n\n""LMI, FMI, and RMR Overview:\n"f"Lean Mass Index (LMI): {metrics['LMI']} kg/m² (Low: <18, Average: 18-20, Fitness: 20-23, Athletic: 23-26)\n"f"Fat Mass Index (FMI): {metrics['FMI']} kg/m² (Very Low: <2, Healthy: 3-6, High: 6-9, Very High: >9)\n"f"Resting Metabolic Rate (RMR): {metrics['RMR']} kcal/day ({metrics['RMR']*7:.0f} kcal/week)\n\n""Now, let’s get to your goal. Based on your current metrics:\n"f"- Height: {metrics['Height']} cm\n"f"- Weight: {metrics['Weight']} kg\n"f"- Body Fat Percentage: {metrics['BFP']}%\n""Please provide a detailed goal for the user to achieve a healthier body composition, considering their current metrics. Include specific recommendations, the amount of fat to shed or gain, the target weight, estimated time to achieve the goal, and the required calorie deficit or surplus.\n\n""Goal Progression:\n""Provide a progress graph of how the user's fat level will change month by month, along with their weight at each point. Include the total energy burned or gained during this process.\n\n""Nutrition Summary:\n""Provide a detailed daily nutrition plan based on the user's goals, including daily calorie intake, proteins, fats, and carbs. Mention things to do and avoid during this nutrition plan.\n\n""Fitness Summary:\n""Outline a fitness plan including workout types, intensity, duration, and frequency per week. Include specific actions to take and things to avoid.\n"
    )

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a fitness and health expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.5
    )

    return response.choices[0].message.content.strip()
