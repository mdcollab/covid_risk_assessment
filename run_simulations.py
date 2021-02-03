import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

from ..risk_simulation.covid_simulation import CovidSimulation
from ..plotting_utils.plotting_utils import add_caption
from ..utils.utils import get_percent


DEFAULT_CONFIG = {
    'infection_to_detectable_delay': 0,
    'beta': 0.2,
    'gamma': 0.07,
    'Q_duration': 14,
    'I_initial': 5,
    'N': 200,
    'num_days': 130,
    'external_infection_rate': 0.001,
}


def run_simulations(config, niter=100):
    all_state_counts = []
    all_cumulative_infections = []
    for i in range(niter):
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


def simulate_pcr_vs_antigen(config):
    config = {**config, **DEFAULT_CONFIG}

    config_types = [
        'No testing', 
        'PCR every 4 days', 
        'Antigen every 4 days', 
        'No intra-office infections',
    ]
    configs = [config.copy() for _ in range(len(config_types))]

    # No testing parameters
    configs[0]['testing_interval'] = None

    # PCR parameters
    configs[1]['testing_delay'] = 5
    configs[1]['sensitivity'] = 0.98

    # Antigen parameters
    configs[2]['testing_delay'] = 0
    configs[2]['sensitivity'] = 0.75

    # No intra-office infections parameters
    configs[3]['beta'] = 0

    return {
        info: run_simulations(config) for info, config in zip(
            config_types, configs
        )
    }

def simulate_pcr_vs_antigen_weekly(config):
    config = {**config, **DEFAULT_CONFIG}

    config_types = [
        '40% Antigen weekly', 
        '75% Antigen weekly', 
        '40% Antigen twice weekly', 
        '75% Antigen twice weekly', 
        '98% PCR weekly',
    ]
    configs = [config.copy() for _ in range(len(config_types))]    

    # Antigen parameters
    configs[0]['testing_delay'] = 0
    configs[0]['sensitivity'] = 0.4  # 0.75
    configs[0]['testing_interval'] = 7

    # Antigen parameters 
    configs[1]['testing_delay'] = 0
    configs[1]['sensitivity'] = 0.75  # 0.75
    configs[1]['testing_interval'] = 7

    # Antigen parameters
    configs[2]['testing_delay'] = 0
    configs[2]['sensitivity'] = 0.4  # 0.75
    configs[2]['testing_interval'] = 3.5

    # Antigen parameters 
    configs[3]['testing_delay'] = 0
    configs[3]['sensitivity'] = 0.75  # 0.75
    configs[3]['testing_interval'] = 3.5

    cadences = [None, None, None, None, 7]

    # PCR parameters
    for i in range(4, len(config_types)):
        configs[i]['testing_delay'] = 5
        configs[i]['sensitivity'] = 0.98
        configs[i]['testing_interval'] = cadences[i]

    return {
        info: run_simulations(config) for info, config in zip(
            config_types, configs
        )
    }


def simulate_testing_candence(config):
    config = {**config, **DEFAULT_CONFIG}

    config_types = [
        'No testing', 
        'Testing every 2 weeks', 
        'Testing every week', 
        'Testing every 4 days',
        'Testing every 3 days',
        'Testing every 2 days',
        'Testing everyday',
    ]
    configs = [config.copy() for _ in range(len(config_types))]

    for i, value in enumerate([None, 14, 7, 4, 3, 2, 1]):
        configs[i]['testing_interval'] = value

    return {
        info: run_simulations(config) for info, config in zip(
            config_types, configs
        )
    }


def quantify_difference(cumulative_infs, result):
    last_ind = len(cumulative_infs) - 1
    percent = get_percent(cumulative_infs[last_ind], DEFAULT_CONFIG['N'])
    print(
        f'For {result}, {percent:.2f}% of individuals were infected at the end '
        'of the n-days'
    )


def simulate_testing_process(config, include_asy=False):
    config = {**config, **DEFAULT_CONFIG}

    config_types = ['Random selection', 'Symptomatic first']
    if include_asy:
        config_types += ['Asymptomatic first']

    configs = [config.copy() for _ in range(len(config_types))]

    configs[0]['testing_process'] = 'random'
    configs[1]['testing_process'] = 'sym_first'
    if include_asy:
        configs[2]['testing_process'] = 'asy_first'

    return {
        info: run_simulations(config) for info, config in zip(
            config_types, configs
        )
    }


def simulate_risk_behavior(config):
    config = {**config, **DEFAULT_CONFIG}

    config_types = [
        '0% with symptoms stay home', 
        '25% with symptoms stay home', 
        '50% with symptoms stay home', 
        '75% with symptoms stay home', 
        '100% with symptoms stay home', 
    ]
    configs = [config.copy() for _ in range(len(config_types))]

    configs[0]['risk_behavior'] = 0.0
    configs[1]['risk_behavior'] = 0.25
    configs[2]['risk_behavior'] = 0.5
    configs[3]['risk_behavior'] = 0.75
    configs[4]['risk_behavior'] = 1.0

    return {
        info: run_simulations(config) for info, config in zip(
            config_types, configs
        )
    }


def plot_cumulative_infections(results):
    colors = [
        'black', 'coral', 'royalblue', 'orange', 'gray', 'darkred', 'purple'
    ]
    for result, color in zip(results, colors):
        cumulative_infs = results[result]['cumulative_infections']
        quantify_difference(cumulative_infs, result)

        plt.plot(cumulative_infs, color=color, alpha=1.0, label=result,)

    plt.xlabel('Day', fontsize=14)
    plt.ylabel('Cumulative infections', fontsize=14)
    sns.despine()
    plt.legend()


def get_config_info(config, keys):
    return ', '.join([f'{x}={config[x]}' for x in keys])
        

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'plots/simulations/{filename}', facecolor='white')
    plt.close()


def run_all_simulations(sym_pos_rate=0.65):
    default_config = {
        'sym_pos_rate': sym_pos_rate,
        'testing_interval': 4,
        'testing_process': 'random',
        'risk_behavior': 0,
    }

    sr_str = int(sym_pos_rate * 100)  # to be used for filenames
    
    '''
    ###############################
    ### PCR vs Antigen ############
    ###############################
    for rb in [0.4]: # [0.0, 0.4, 0.5, 0.6, 0.9]:
        print(f'PCR vs Antigen ---- {rb}')
        config_1 = {
            **default_config, **{'testing_delay': 3, 'sensitivity': 0.8,},
        }
        config_1['risk_behavior'] = rb
        results = simulate_pcr_vs_antigen(config_1)
        plot_cumulative_infections(results)
        caption = get_config_info(
            config_1, ['testing_interval', 'testing_delay', 'sensitivity'],
        )
        if rb > 0.0:
            plt.title(f'Risk Behavior={rb}')

        add_caption(caption, locx=0.1, locy=-0.0)
        save_plot(f'pcr_vs_antigen_{int(rb * 100)}_{sr_str}_sr')
        plt.close()
        print('------------------------------------')

    '''
    ###############################
    ### PCR vs Antigen # 2 ########
    ###############################

    config_1 = default_config
    results = simulate_pcr_vs_antigen_weekly(config_1)

    plot_cumulative_infections(results)
    save_plot(f'pcr_vs_antigen_weekly_{sr_str}_sr')
    plt.close()
    print('------------------------------------')
    

    '''    
    ###############################
    ### Testing Cadence ###########
    ###############################
    print('Testing Cadence')
    config_1 = {
        **default_config, **{'testing_delay': 0, 'sensitivity': 0.75,},
    }

    results = simulate_testing_candence(config_1)
    plot_cumulative_infections(results)
    caption = get_config_info(
        config_1, 
        ['testing_delay', 'sensitivity', 'risk_behavior', 'sym_pos_rate'],
    )
    add_caption(caption, locx=0.1, locy=-0.01)
    save_plot(f'testing_cadence_{sr_str}_sr')
    print('------------------------------------')
 
    ###############################
    ### Testing Process ###########
    ###############################
    config_1 = {
         **default_config, **{'testing_delay': 0, 'sensitivity': 0.75,},
    }

    for rb in [0.0, 0.5, 0.9, 1.0]:
        print(f'Testing process ---- {rb}')
        config_1['risk_behavior'] = rb
        results = simulate_testing_process(config_1, include_asy=True)
        plot_cumulative_infections(results)
        caption = get_config_info(
            config_1, 
            [
                'testing_interval', 
                'testing_delay', 
                'sensitivity', 
                'sym_pos_rate',
            ],
        )
        add_caption(caption, locx=0.1, locy=-0.01)
        plt.title(f'Risk Behavior={rb}')
        save_plot(f'testing_process_rb_{int(rb * 100)}_{sr_str}_sr')
        print('------------------------------------')
    '''  
    ###############################
    ### Risk Behavior #############
    ###############################
    config_1 = {
         **default_config, **{'testing_delay': 0, 'sensitivity': 0.75,},
    }

    for tp in ['random', 'sym_first']:
        print(f'Risk Behavior ---- {tp}')
        config_1['testing_process'] = tp
        results = simulate_risk_behavior(config_1)
        plot_cumulative_infections(results)
        caption = get_config_info(
            config_1, 
            [
                'testing_interval', 
                'testing_delay', 
                'sensitivity', 
                'testing_process',
            ],
        )
        add_caption(caption, locx=0.1, locy=-0.001)
        save_plot(f'risk_behavior_{tp}_{sr_str}_sr')
        print('------------------------------------')


if __name__ == "__main__":
    run_all_simulations(sym_pos_rate=0.65)
