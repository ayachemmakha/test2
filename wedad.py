import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("model.pkl")

# Configuration de la page
st.set_page_config(page_title="Prédicteur Publicité Réseau Social", layout="centered")

# Titre
st.title("📱 Prédiction d'Achat via Publicité Réseau Social")
st.markdown("Entrez les informations d’un utilisateur pour prédire s’il achètera le produit ou non.")

# Formulaire utilisateur
with st.form("user_input_form"):
    gender = st.selectbox("Genre", ("Homme", "Femme"))
    age = st.slider("Âge", 18, 70, 30)
    salary = st.number_input("Salaire estimé (€)", min_value=10000, max_value=150000, value=40000, step=5000)
    
    submitted = st.form_submit_button("Prédire")

# Encoder le genre et faire une prédiction
if submitted:
    gender_encoded = 1 if gender == "Homme" else 0
    input_data = np.array([[age, salary]])

    # Si tu as normalisé les données avant l'entraînement
    scaler = joblib.load("scaler.pkl") if "scaler.pkl" in model.__dict__ else None
    if scaler:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    # Résultat
    if prediction == 1:
        st.success("✅ L'utilisateur **achètera** probablement le produit.")
    else:
        st.warning("❌ L'utilisateur **n’achètera pas** le produit.")

    if proba is not None:
        st.markdown(f"**Probabilité d'achat :** {proba:.2%}")

# Pied de page
st.markdown("---")
st.caption("📊 Projet réalisé avec Streamlit, Scikit-learn et Python")
