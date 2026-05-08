import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DUKE_LOGO_PATH = "/Users/eugeniatate/dukelogo.png"

DUKE_BLUE = "#012169"

st.set_page_config(page_title="🏀CameronFair AI", layout="wide")

st.markdown(
    f"""
    <style>
    h1, h2, h3 {{
        color: {DUKE_BLUE};
    }}
    div.stButton > button {{
        background-color: {DUKE_BLUE};
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        border: none;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

try:
    st.image(DUKE_LOGO_PATH, width=250)
except Exception:
    st.write("🏀")

st.title("🏀 CameronFair AI")
st.subheader("Duke Basketball Ticket Lottery")

num_students = st.slider("Number of Students in Lottery Pool", 50, 500, 200)

total_tickets = st.slider("Total Student Tickets Available", 20, 200, 100)

campout_tickets = st.slider(
    "Campout Reserved Tickets",
    0,
    total_tickets,
    50
)

max_lottery_tickets = total_tickets - campout_tickets

lottery_tickets = st.slider(
    "AI Lottery Tickets",
    0,
    max_lottery_tickets,
    max_lottery_tickets
)

display_cols = {

    "name": "Student",

    "predicted_attendance_prob": "Predicted Attendance",

    "merit_score": "Merit Score",

    "trivia_passed": "Trivia Passed",

    "good_gpa_standing": "Good GPA Standing",

    "volunteer_hours": "Volunteer Hours",

    "duke_event_participation": "Duke Event Participation",

    "student_section_engagement": "Student Section Engagement",

    "past_attendance_rate": "Past Attendance Rate",

    "no_show_rate": "No-Show Rate"

}

st.info(
    f"Total tickets: {total_tickets} | "
    f"Campout reserved: {campout_tickets} | "
    f"AI lottery: {lottery_tickets}"
)

if st.button("Run Allocation"):

    np.random.seed(42)

    students = pd.DataFrame({
        "name": [f"Student_{i}" for i in range(num_students)],

        # Eligibility / merit features
        "trivia_passed": np.random.choice([0, 1], size=num_students, p=[0.25, 0.75]),
        "good_gpa_standing": np.random.choice([0, 1], size=num_students, p=[0.15, 0.85]),
        "volunteer_hours": np.random.randint(0, 80, num_students),
        "duke_event_participation": np.random.randint(0, 100, num_students),
        "student_section_engagement": np.random.randint(0, 100, num_students),

        # Reliability features
        "past_attendance_rate": np.random.uniform(0.4, 1.0, num_students),
        "no_show_rate": np.random.uniform(0.0, 0.35, num_students)
    })

    students["volunteer_score"] = students["volunteer_hours"] / students["volunteer_hours"].max()
    students["event_score"] = students["duke_event_participation"] / 100
    students["engagement_score"] = students["student_section_engagement"] / 100

    students["merit_score"] = (
        0.25 * students["trivia_passed"] +
        0.20 * students["good_gpa_standing"] +
        0.20 * students["volunteer_score"] +
        0.15 * students["event_score"] +
        0.20 * students["engagement_score"]
    )

    students["true_attendance_likelihood"] = (
        0.35 * students["past_attendance_rate"] +
        0.30 * students["merit_score"] +
        0.20 * students["trivia_passed"] +
        0.15 * students["good_gpa_standing"] -
        0.35 * students["no_show_rate"]
    )

    students["actual_show"] = (
        students["true_attendance_likelihood"] + np.random.normal(0, 0.08, num_students)
    ) > 0.55

    students["actual_show"] = students["actual_show"].astype(int)

    features = [
        "trivia_passed",
        "good_gpa_standing",
        "volunteer_score",
        "event_score",
        "engagement_score",
        "past_attendance_rate",
        "no_show_rate",
        "merit_score"
    ]

    X = students[features]
    y = students["actual_show"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    students["predicted_attendance_prob"] = model.predict_proba(X)[:, 1]

    eligible_students = students[
        (students["trivia_passed"] == 1) &
        (students["good_gpa_standing"] == 1)
    ].copy()

    if lottery_tickets > len(eligible_students):
        lottery_tickets = len(eligible_students)

    weights = (
        eligible_students["predicted_attendance_prob"] *
        eligible_students["merit_score"]
    )

    weights = weights / weights.sum()

    winners = eligible_students.sample(
        n=lottery_tickets,
        weights=weights,
        random_state=1
    ).copy()

    winners["show_up"] = (
        np.random.rand(len(winners)) < winners["predicted_attendance_prob"]
    )

    no_shows = winners[winners["show_up"] == False].copy()

    remaining_students = eligible_students[
        ~eligible_students["name"].isin(winners["name"])
    ].copy()

    if len(no_shows) > 0 and len(remaining_students) > 0:
        replacement_count = min(len(no_shows), len(remaining_students))

        replacement_weights = (
            remaining_students["predicted_attendance_prob"] *
            remaining_students["merit_score"]
        )

        replacement_weights = replacement_weights / replacement_weights.sum()

        replacements = remaining_students.sample(
            n=replacement_count,
            weights=replacement_weights,
            random_state=2
        ).copy()
    else:
        replacements = pd.DataFrame(columns=students.columns)

    st.markdown("## Ticket Winners")

    st.dataframe(
         winners[display_cols.keys()]
        .rename(columns=display_cols)
        .sort_values("Predicted Attendance", ascending=False),
        use_container_width=True
    )

    st.markdown("## Simulated No-Shows")

    if len(no_shows) == 0:
        st.success("No simulated no-shows.")
    else:
        st.dataframe(
            no_shows[[
                "name",
                "predicted_attendance_prob",
                "merit_score",
                "no_show_rate"
            ]],
            use_container_width=True
        )

    st.markdown("## Replacement Winners")

    if len(replacements) == 0:
        st.info("No replacements needed.")
    else:
        st.dataframe(
            replacements[display_cols.keys()]
            .rename(columns=display_cols)
            .sort_values("Predicted Attendance", ascending=False),
            use_container_width=True
        )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)

    st.markdown("## What Features Drive the Model?")

    st.dataframe(
        feature_importance,
        use_container_width=True
    )