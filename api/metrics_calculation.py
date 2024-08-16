import math

def calculate_final_metrics(sex, neck_circumference, waist_circumference, hip_circumference, height, weight):

    def calculate_bfp(sex, neck_circumference, waist_circumference, hip_circumference, height):
        """
        Calculate Body Fat Percentage using the Navy Body Fat formula.
        """
        if sex.lower() == 'male':
            bfp = 86.010 * math.log10(waist_circumference - neck_circumference) - 70.041 * math.log10(height) + 36.76
        elif sex.lower() == 'female':
            bfp = 163.205 * math.log10(waist_circumference + hip_circumference - neck_circumference) - 97.684 * math.log10(height) - 78.387
        else:
            raise ValueError("Sex must be 'male' or 'female'")
        return bfp

    def classify_fat_types(weight, bfp):
        """
        Classify fat types into essential, beneficial, and unbeneficial fats.
        """
        fm = weight * (bfp / 100)
        if sex.lower() == 'male':
            essential_fat = max(0.02 * weight, min(0.05 * weight, fm))
        elif sex.lower() == 'female':
            essential_fat = max(0.10 * weight, min(0.13 * weight, fm))
        beneficial_fat = 0.15 * weight  # 15% of total body weight for beneficial fat
        unbeneficial_fat = fm - (essential_fat + beneficial_fat)
        return essential_fat, beneficial_fat, unbeneficial_fat

    def calculate_lean_mass(weight, fm):
        """
        Calculate Lean Mass.
        """
        return weight - fm

    def calculate_indices(lean_mass, fm, height):
        """
        Calculate Lean Mass Index (LMI) and Fat Mass Index (FMI).
        """
        height_m = height / 100  # convert height to meters
        lmi = lean_mass / (height_m ** 2)
        fmi = fm / (height_m ** 2)
        return lmi, fmi

    def calculate_rmr(lean_mass):
        """
        Calculate Resting Metabolic Rate (RMR) using the Katch-McArdle formula.
        """
        return 370 + (21.6 * lean_mass)

    # Calculations
    bfp = calculate_bfp(sex, neck_circumference, waist_circumference, hip_circumference, height)
    fm = weight * (bfp / 100)
    essential_fat, beneficial_fat, unbeneficial_fat = classify_fat_types(weight, bfp)
    lean_mass = calculate_lean_mass(weight, fm)
    lmi, fmi = calculate_indices(lean_mass, fm, height)
    rmr = calculate_rmr(lean_mass)

    metrics = {
        'Neck': neck_circumference,
        'Waist': waist_circumference,
        'Hip': hip_circumference,
        'Height': height,
        'Weight': weight,
        'BFP': bfp,
        'Essential_Fat': essential_fat,
        'Beneficial_Fat': beneficial_fat,
        'Unbeneficial_Fat': unbeneficial_fat,
        'Lean_Mass': lean_mass,
        'LMI': lmi,
        'FMI': fmi,
        'RMR': rmr
    }

    # Print Results
    print(f"Body Fat Percentage (BFP): {bfp:.2f}%")
    print(f"Essential Fat: {essential_fat:.2f} kg")
    print(f"Beneficial Fat: {beneficial_fat:.2f} kg")
    print(f"Unbeneficial Fat: {unbeneficial_fat:.2f} kg")
    print(f"Lean Mass: {lean_mass:.2f} kg")
    print(f"Lean Mass Index (LMI): {lmi:.2f} kg/m^2")
    print(f"Fat Mass Index (FMI): {fmi:.2f} kg/m^2")
    print(f"Resting Metabolic Rate (RMR): {rmr:.2f} kcal/day")

    return metrics