import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.title("🏀 CameronFair AI")
st.subheader("AI-Powered Duke Basketball Ticket Allocation")

# Controls
num_students = st.slider("Number of Students", 10, 100, 30)
num_tickets = st.slider("Tickets Available", 5, 30, 10)

if st.button("Run Allocation"):

    np.random.seed(42)

    # Generate data
    students = pd.DataFrame({
        "name": [f"Student_{i}" for i in range(num_students)],
        "engagement": np.random.randint(1, 100, num_students),
        "past_attendance": np.random.rand(num_students)
    })

    # Simulate historical attendance
    students["attendance_prob"] = 0.5 * (students["engagement"]/100) + 0.5 * students["past_attendance"]

    students["actual_show"] = (
        students["attendance_prob"] + np.random.normal(0, 0.1, len(students))
    ) > 0.5
    students["actual_show"] = students["actual_show"].astype(int)

    # Train model
    X = students[["engagement", "past_attendance"]]
    y = students["actual_show"]

    model = LogisticRegression()
    model.fit(X, y)

    # Predict attendance
    students["predicted_prob"] = model.predict_proba(X)[:, 1]

    # AI Allocation
    weights = students["predicted_prob"] / students["predicted_prob"].sum()
    winners = students.sample(n=num_tickets, weights=weights, random_state=1)

    st.subheader("🎟️ AI Ticket Winners")
    st.dataframe(winners[["name", "predicted_prob"]])

    # Simulate no-shows (AI)
    winners["show_up"] = np.random.rand(len(winners)) < winners["predicted_prob"]
    no_shows = winners[winners["show_up"] == False]

    st.subheader("❌ AI No-Shows")
    st.write(no_shows["name"].tolist())

    # Random baseline
    random_winners = students.sample(n=num_tickets, random_state=1)
    random_winners["show_up"] = np.random.rand(len(random_winners)) < random_winners["predicted_prob"]
    random_no_shows = random_winners[random_winners["show_up"] == False]

    st.subheader("📊 Comparison")
    st.write(f"🎲 Random No-Shows: {len(random_no_shows)}")
    st.write(f"🤖 AI No-Shows: {len(no_shows)}")

    # Reallocation
    remaining = students[~students["name"].isin(winners["name"])]
    new_weights = remaining["predicted_prob"] / remaining["predicted_prob"].sum()
    new_winners = remaining.sample(n=len(no_shows), weights=new_weights)

    st.subheader("🔁 Reallocated Tickets")
    st.write(new_winners["name"].tolist())

    # Visualization
    st.subheader("📈 AI Predicted Attendance")
    st.bar_chart(students.set_index("name")["predicted_prob"])