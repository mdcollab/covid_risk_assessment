import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import time

from covid_simulation import CovidSimulation

NITER = 25

def quantify_infected(cumulative_infs, result, config):
    last_ind = len(cumulative_infs) - 1
    percent = (cumulative_infs[last_ind] / config['N']) * 100

    return percent


def plot_st_chart(data, var_name='Policy', value_name=''):
    data['Day'] = data.index
    data = data.melt('Day', var_name=var_name, value_name=value_name)
    plot_container.altair_chart(
        alt.Chart(data).mark_line().encode(
            x="Day", y=value_name, color=var_name,
        ).interactive(), 
        use_container_width=True,
    )

def plot_cumulative_infections(results, config):
    best_result = float('inf')
    for config_type in results:
        cumulative_infs = results[config_type]['cumulative_infections']
        percent_infected = quantify_infected(
            cumulative_infs, config_type, config
        )

        if percent_infected < best_result:
            best_result = percent_infected
            best_config_type = config_type

    # Plotting `Cumulative Infections` plot
    plot_container.subheader('Number of Cumulative Infections')
    to_plot = pd.DataFrame(
        {k: v['cumulative_infections'] for k, v in results.items()}
    )
    plot_st_chart(to_plot, value_name='Cumulative Infections Count')
       
    # Plotting `Number Quarantining` plot
    plot_container.subheader('Number of Employees Quarantining Per Day')
    to_plot_q = pd.DataFrame(
        {k: v['state_counts']['Q'] for k, v in results.items()}
    )
    plot_st_chart(to_plot_q, value_name='Quarantining Count')

    # Printing results info
    plot_container.subheader('Results')
    plot_container.write(
        f'The best option for the business is: {best_config_type}.'
    )
    plot_container.write('')


def run_simulations(
    config, niter=NITER, bar=None, n_configs=1, i_config=1, placeholder_iter=None,
):
    all_state_counts = []
    all_cumulative_infections = []
    for i in range(niter):
        full_progress_fraction = 1 / n_configs 
        bar.progress(
            i_config * full_progress_fraction + (
                i * full_progress_fraction / niter
            )
        )
        placeholder_iter.text(f'On iteration {i + 1} of {NITER}')

        state_counts, cumulative_infections, _, _ = (
            CovidSimulation(**config).run_simulation()
        )
        all_state_counts.append(state_counts)
        all_cumulative_infections.append(cumulative_infections)
    
    return {
        'state_counts': pd.concat(all_state_counts).groupby(level=0).mean(),
        'cumulative_infections': (
            pd.concat(all_cumulative_infections).groupby(level=0).mean()
        ),
    }


def run_simulation(config):
    config_types = [
        'PCR only',
        'Antigen only',
        '50-50',
        'Symptom dependent',
    ]
    configs = [config.copy() for _ in range(len(config_types))]    
 
    configs[0]['test_type_process'] = 'all_pcr'
    configs[1]['test_type_process'] = 'all_antigen'
    configs[2]['test_type_process'] = '50_50'
    configs[3]['test_type_process'] = 'sym_dependent'

    # Add a placeholder
    placeholder_config = st.empty()
    placeholder_iter = st.empty()
    bar = st.progress(0)

    results = {}
    i = 0
    for info, config in zip(config_types, configs):
        # Update the progress bar with each iteration.
        placeholder_config.text(f'Simulating policy: {info}')

        n_configs = len(config_types)
        bar.progress(i / n_configs)
        results[info] = run_simulations(
            config, 
            bar=bar, 
            n_configs=n_configs,
            i_config=i,
            placeholder_iter=placeholder_iter,
        )

        i += 1
    bar.progress(i * (1 / len(config_types)))

    return results


DEFAULT_CONFIG = {
    'infection_to_detectable_delay': 0,
    'gamma': 0.07,
    'Q_duration': 14,
    'I_initial': 3,
    'num_days': 130,
    'external_infection_rate': 0.001,
}

st.write("# COVID Testing Policy Planning for Outbreak Reduction")

st.sidebar.header("Data")

######################################
######################################
# Side Bar ###########################
######################################
######################################

######################################
# N Parameter ########################
######################################

N = st.sidebar.text_input('Number of employees:', 5)

if not N.isnumeric():
    st.sidebar.error("Error: this should be a numeric value.")

######################################
# Beta Parameter #####################
######################################

sd_info = st.sidebar.selectbox(
    "Does the workplace allow for 6 feet of distance?", ('Yes', 'No'),
)

sd_hours = st.sidebar.slider(
    "Around how many hours a day are employees less than 6 feet apart?", 0, 12
)

mask_info = st.sidebar.selectbox(
    "Are masks required?",
    (
        'Required', 
        'Advised, but not required', 
        'Not allowed (i.e. it interferes with the job requirements)'
    ),
)
share_info = st.sidebar.selectbox(
    "Do employees share common spaces/conference rooms/same tools?",
    ('Yes', 'No'),
)

share_hours = st.sidebar.slider(
    "Around how many hours a day are employees in the same shared space?", 0, 12
)

if sd_info == 'No':
    sd_beta = 0.4
elif sd_info == 'Yes':
    sd_beta = 0.3

MASK_DECREASE = 0.21

if mask_info == 'Required':
    mask_beta = MASK_DECREASE
elif mask_info == 'Advised, but not required':
    mask_beta = MASK_DECREASE / 2
else:
    mask_beta = 0

if share_info == 'Yes':
    share_beta = 0.1
else:
    share_beta = 0

MAX_BETA = 0.8
beta = (MAX_BETA * sd_hours + share_beta) * mask_beta 

######################################
# R Parameter ########################
######################################

R = st.sidebar.text_input('Number of vaccinated employees:', 0)

if R > N:
    st.sidebar.error(
        "Error: you cannot have more vaccinated employees than total employees."
    )

######################################
# Risk Behavior Parameter ############
######################################

sl_policy = st.sidebar.selectbox(
    "What is your sick leave policy?", 
    ('Paid sick leave', 'Unpaid sick leave', 'Limited sick leave allowed'),
)
rw_policy = st.sidebar.selectbox(
    "Is remote work allowed?", 
    ('Yes', 'Allowed under special cirsumstanes', 'Not allowed'),
)

if sl_policy == 'Paid sick leave':
    sl_score = 0.6 
elif sl_policy == 'Unpaid sick leave':
    sl_score = 0.2
elif sl_policy == 'Limited sick leave allowed':
    sl_score = 0.1

if rw_policy == 'Yes':
    risk_score = 0.2 
elif rw_policy == 'Not allowed':
    risk_score = -0.1
else:
    risk_score = 0

risk_behavior = sl_score + risk_score

######################################
# Testing Cadence Parameter ##########
######################################

testing_interval = st.sidebar.text_input(
    'At what interval (in days) do you want employees tested:', 1
)

if not N.isnumeric():
    st.sidebar.error("Error: this should be a numeric value.")

######################################
######################################
# Main Screen ########################
######################################
######################################

indent = '&nbsp;&nbsp;&nbsp;&nbsp;'

######################################
# Instructions #######################
######################################

st.write("")
st.header("Instructions")
st.write("")

st.markdown(
    "1. Input data values in the side bar titled 'Data'. This will set the "
    "parameters values that are inputted into the simulation model."
)
st.markdown("2. Click 'Run Simulation' button below.")
st.markdown(
    "3. After simulation runs, results and plots will be plotted below."
)

######################################
# Parameters #########################
######################################

st.markdown(
    """
    <style>
    .small-font {
        font-size:12px !important;
    }
    </style>
    """, unsafe_allow_html=True
)

st.write("")
st.header("Parameters")
st.write("")

st.write("The total number of employees.")
st.markdown(f"- {indent}N={N}")

st.write(
    "The likelihood of contracting COVID-19 in the office given an infected "
    "case (between 0-1)."
)

st.markdown(
    """
    <p class="small-font">
        This will be affected by how socially-distanced employees are in the
        workplace; if masks are required in the workplace; and if employees 
        share common spaces/conference/same tools.
    </p>
    """, unsafe_allow_html=True
)

st.markdown(f"- {indent}beta={beta:.2f}")

st.write(
    "The percentage of employees who are likely to stay home given onset of "
    "symptoms."
)
st.markdown(
    """
    <p class="small-font">
        This will be affected by sick leave policy and by remote work ability.
    </p>
     """, unsafe_allow_html=True
)
st.markdown(f"- {indent}risk_behavior={risk_behavior:.2f}")

st.write("The cadence at which employees are tested.")
st.markdown(f"- {indent}testing_interval={int(testing_interval):.2f}")

######################################
# Simulation #########################
######################################

st.write("")
st.header("Simulation")

config = {
    **DEFAULT_CONFIG, 
    **{
        'N': int(N), 
        'R_initial': int(R), 
        'beta': beta, 
        'testing_interval' : int(testing_interval),
        'risk_behavior': risk_behavior,
    }
}

plot_container = st.beta_container()

if st.button('Run Simulation'):
    start_time = time.time()
    results = run_simulation(config)
    plot_cumulative_infections(results, config)
    
    end_time = time.time()

    seconds = end_time - start_time
    st.text(f'Simulations took {seconds / 60:.2f} minutes to run.')
