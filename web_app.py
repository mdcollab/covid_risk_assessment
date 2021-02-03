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
        'Antigen - Symptomatic first',
        'Antigen - Asymptomatic first',
        'PCR - Symptomatic first',
        'PCR - Asymptomatic first',
    ]
    configs = [config.copy() for _ in range(len(config_types))]    

    antigen_config = {**config, **{'sensitivity': 0.75, 'testing_delay': 0}}
    pcr_config = {**config, **{'sensitivity': 0.98, 'testing_delay': 5}}
 
    configs[0] = antigen_config
    configs[0]['testing_process'] = 'sym_first'
    configs[1] = antigen_config
    configs[1]['testing_process'] = 'asy_first'
    configs[2] = pcr_config
    configs[2]['testing_process'] = 'sym_first'
    configs[3] = pcr_config
    configs[3]['testing_process'] = 'asy_first'

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
    "How socially distanced are employees in the office?",
    ('Not 6 feet apart', 'Sometimes 6 feet apart', 'Everyone 6 feet apart'),
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
    "Do employees share common spaces/conference rooms/same tools??",
    ('Yes', 'No'),
)

if sd_info == 'Not 6 feet apart':
    sd_beta = 0.4
elif sd_info == 'Sometimes 6 feet apart':
    sd_beta = 0.3
elif sd_info == 'Everyone 6 feet apart':
    sd_beta = 0.2

if mask_info == 'Required':
    mask_beta = -0.1
elif mask_info == 'Not allowed (i.e. it interferes with the job requirements)':
    mask_beta = 0.1
else:
    mask_beta = 0

if share_info == 'Yes':
    share_beta = 0.1
else:
    share_beta = 0

beta = sd_beta + mask_beta + share_beta

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

num_tests = st.sidebar.text_input(
    'How many employees do you want to test per day:', 1
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

st.write("The number of tests available per day.")
st.markdown(f"- {indent}num_tests_daily={int(num_tests):.2f}")

######################################
# Simulation #########################
######################################

st.header("Simulation")

config = {
    **DEFAULT_CONFIG, 
    **{
        'N': int(N), 
        'R_initial': int(R), 
        'beta': beta, 
        'num_tests_daily': int(num_tests),
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
