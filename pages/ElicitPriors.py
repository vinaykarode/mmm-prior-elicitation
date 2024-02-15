import streamlit as st
import pymc as pm
# Import custom functions
from prior_functions import *


# Give some context for what the page displays
st.title("Elicit Priors for Ad Channel CPA")

# Step 1: User inputs for the confidence interval and mass
st.markdown("### Please provide your estimates:")
MASS = st.slider("I'm   **:blue[[blank]] %**   confident:", min_value=80, max_value=95, value=90, step=1)
LOWER = st.number_input(f"I'm {MASS}% confident we need to spend at least **:blue[[blank]]** euros to get one extra sale:", min_value=0, value=300)
UPPER = st.number_input(f"I'm {MASS}% confident we do not need to spend more than **:blue[[blank]]** euros to get one extra sale:", min_value=0, value=1500)
MASS = MASS / 100 # convert from %

# Step 2: Display the user's confidence interval and mass in markdown
st.divider()
st.markdown(f"**:gray[I'm :blue[_{MASS*100:.0f}_%] sure we need to spend between :blue[_€{LOWER}_] and :blue[_€{UPPER}_] on this ad channel to get one additional sale.]**")
st.divider()

# Button to calculate and plot the distribution
if st.button('Calculate and Plot Optimal Gamma Distribution'):
    if LOWER >= UPPER:
        st.error("The lower bound must be less than the upper bound. Please adjust your inputs.")
    else:
        # Step 3: Calculate constrained priors and plot
        constrained_priors = find_optimal_gamma_parameters(LOWER, UPPER, MASS)
        draws = draw_samples_from_prior(dist=pm.Gamma, **constrained_priors)

        fig = plot_prior_distribution(draws, title="Optimal Gamma Distribution for Ad Channel CPA")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
                    :green[**Optimal alpha $$\\alpha$$**:] {round(constrained_priors['alpha'], 3)}\n
                    :green[**Optimal beta $$\\beta$$**:] {round(constrained_priors['beta'], 3)}
                    """)
