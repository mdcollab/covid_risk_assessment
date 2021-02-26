import math
import numpy as np
import pandas as pd

class CovidSimulation():

    def __init__(
        self,
        testing_interval=4,
        num_tests=None,
        infection_to_detectable_delay=0,
        beta=0.2, 
        gamma=0.07, 
        Q_duration=14,
        R_initial=0,
        I_initial=5,
        N=200,
        num_days=130,
        sensitivity_antigen_sym=0.85,
        sensitivity_antigen_asy=0.4,
        sensitivity_pcr=0.98,
        delay_antigen=0,
        delay_pcr=5,
        test_type_process='all_pcr',
        external_infection_rate=0.001,
        risk_behavior=0.7,
        testing_process='sym_first',
        sym_pos_rate=0.65,
        sym_neg_rate=0.05,
    ):
        """Initialize `self.population` with every state as `S` (Susceptible).
        
        Parameters
        ----------
        testing_interval: 
            The frequency at which to test all employees.
        num_tests:
            The number of tests available during testing days. If None, then 
            this value to be equal to N.
        infection_to_detectable_delay:
            The number of days between infection and when it would be detectable 
            in a test.
        beta:
            The number of contacts per person that would result in an infection 
            (if one was S and the other I).
        gamma:
            Recovery rate.
        Q_duration:
            Quarantine duration.
        R_initial:
            Number of employees already recovered (or vaccinated) on day 1.
        I_initial:
            Number of employees already infected on day 1.
        N:
            Number of people within company population.
        num_days:
            Number of days to simulate.
        sensitivity_antigen_sym:
            The sensitivity (aka recall) of the antigen test (true positives / 
            num_infected) for symptomatic cases.
        sensitivity_antigen_asy:
            The sensitivity (aka recall) of the antigen test (true positives / 
            num_infected) for asymptomatic cases.
        sensitivity_pcr:
            The sensitivity (aka recall) of the PCR test (true positives / 
            num_infected) for all cases.
        delay_antigen:
            The number of days between the antigen test and results.
        delay_pcr:
            The number of days between the PCR test and results.
        test_type_process:
            The test type to administer. Options include: `all_pcr`, 
            `all_antigen`, `50_50` (randomly select half and half), 
            `sym_dependent` (antigen for those with symptoms, PCR otherwise).
        external_infection_rate: 
            The probability on any given day that someone comes in 
            infection-causing contact with an infected person outside the 
            population.
        risk_behavior:
            The probability an individual would choose to self-quarantine if 
            they display symptoms.
        testing_process:
            Either `sym_first`, `asy_first`, or `random`.
        sym_pos_rate:
            Symptomatic rate given a COVID-19 positive case.
        sym_neg_rate:
            Symptomatic rate (of COVID-19-like symptoms) given a negative case 
            (i.e. rate of flu or respiratory illnesses among cases without 
            COVID-19).
        """
        self.population = pd.DataFrame(
            {
                'state': 'S',
                'positive_test_dates': [set() for _ in range(N)],
                'negative_test_dates': [set() for _ in range(N)],
                'quarantine_start_date': np.nan,
                'infection_date': np.nan,
                'has_fever': False,
                'is_symptomatic': False,
                'last_tested_date': np.nan,
                'known_to_be_recovered': False,
            }
        )
        
        self.population['id'] = self.population.index
        self.population['known_to_be_recovered'] = False
        
        self.state_logs = []
        self.state_counts = {}
        
        self.testing_interval = testing_interval
        
        if num_tests is None:
            self.num_tests = N
        else:
            self.num_tests = num_tests
        
        self.infection_to_detectable_delay = infection_to_detectable_delay
        self.beta = beta
        self.gamma = gamma
        self.Q_duration = Q_duration
        self.R_initial = R_initial
        self.I_initial = I_initial
        self.N = N
        self.num_days = num_days
        self.sensitivity_antigen_sym = sensitivity_antigen_sym
        self.sensitivity_antigen_asy = sensitivity_antigen_asy
        self.sensitivity_pcr = sensitivity_pcr
        self.delay_antigen = delay_antigen
        self.delay_pcr = delay_pcr
        self.test_type_process = test_type_process
        self.external_infection_rate = external_infection_rate
        self.risk_behavior = risk_behavior
        self.testing_process= testing_process
        self.sym_pos_rate = sym_pos_rate
        self.sym_neg_rate = sym_neg_rate
             
    def introduce_initial_infections(self):
        """Introduce initial infections depending on `self.I_initial`.
        """
        initial_infections = np.random.choice(
            self.population.index, self.I_initial, replace=False
        )

        self.population.loc[initial_infections, 'state'] = 'I'
        self.population.loc[initial_infections, 'infection_date'] = 0 
    
    def introduce_initial_recovered(self):
        """Introduce initial recovered depending on `self.R_initial`.
        """
        initial_recovered = np.random.choice(
            self.population.index, self.R_initial, replace=False
        )

        self.population.loc[initial_recovered, 'state'] = 'R'
        self.population.loc[initial_recovered, 'known_to_be_recovered'] = True

    def log_states(self, day):
        self.state_logs += [self.population['state'].rename(day)]
        
        self.state_counts[day] = (
            self.population['state'].value_counts().to_dict()
        )
        self.S = self.state_counts[day].get('S', 0)
        self.I = self.state_counts[day].get('I', 0)
        self.R = self.state_counts[day].get('R', 0)
        self.Q = self.state_counts[day].get('Q', 0)
    
    def log_updated_states(self, day):                
        self.cumulative_infections = (
            self.population['infection_date']
            .value_counts()
            .sort_index()
            .cumsum()
            .reindex(range(self.num_days), method='pad')
        )
    
    def get_testing_selection(self, day):
        """Get select number of people for testing.
        
        Number of people will depend on `num_tests`, and whether they pass the 
        criteria necessary for selection, which include not having been tested 
        recently (`last_tested_cutoff`).
        """        
        if self.testing_process == 'sym_first':
            criteria_filter = self.population.is_symptomatic == True
        elif self.testing_process == 'asy_first':
            criteria_filter = self.population.is_symptomatic == False
        elif self.testing_process == 'random':
            # no filter - all Trues
            criteria_filter = np.array([True for _ in range(self.N)])
        
        # Get selection of those tested less recently
        ordered_by_last_test = self.population.sort_values(
            'last_tested_date', na_position='first'
        )
        last_tested_cutoff = ordered_by_last_test.iloc[
            self.num_tests_daily - 1, :
        ]['last_tested_date']
        
        # If never tested, `last_tested_date` will be NaN, and therefore False 
        # in the `was_recently_tested` boolean Series.
        was_recently_tested = (
            self.population.last_tested_date > last_tested_cutoff
        )
    
        # Don't test those that recovered after quarantining.
        # Do test those that recovered but did not receive positive test yet. 
        # (i.e. don't know that they were ever infected or that they recovered).
        # But you can test those that are symptomatic and quarantining.      
        pass_criteria = (
            criteria_filter &
            ~self.population.known_to_be_recovered &
            ~was_recently_tested
        )
        choose_from = self.population.loc[pass_criteria, :]
        
        num_tests_left = self.num_tests_daily - sum(pass_criteria)
        # If more pass criteria then tests available, randomly select from those
        # that pass criteria and return selection: 
        if num_tests_left <= 0:
            return np.random.choice(
                choose_from.index, self.num_tests_daily, replace=False
            )
        
        # If less pass criteria then tests available, choose all that pass 
        # criteria plus randomly select from those that don't that haven't been
        # recently tested:
        selection = choose_from.index
        randomly_select_from = (
            self.population[~pass_criteria & ~was_recently_tested]
        )
        
        return np.concatenate(
            (selection, np.random.choice(
                randomly_select_from.index, num_tests_left, replace=False
            ))
        )
    
    def get_test_type_selections(self, selection):
        if self.test_type_process == 'all_antigen':
            selection_antigen = selection
            selection_pcr = np.empty(0)
        elif self.test_type_process == 'all_pcr':
            selection_antigen = np.empty(0)
            selection_pcr = selection
        elif self.test_type_process == '50_50':
            selection_antigen = np.random.choice(
                selection, int(self.N / 2), replace=False
            )
            selection_pcr = [x for x in selection if x not in selection_antigen]
        elif self.test_type_process == 'sym_dependent':
            selection_antigen = [x for x in (
                self.population[self.population.is_symptomatic].index
            ) if x in selection]
            selection_pcr = [x for x in (
                self.population[~self.population.is_symptomatic].index
            ) if x in selection]

        assert len(selection) == (len(selection_antigen) + len(selection_pcr))

        return selection_antigen, selection_pcr

    def run_tests(self, day):
        """Run tests on everyone assigned to test on that date.
        """
        if self.num_tests < self.N:
            selection = self.get_testing_selection(day)
        else:
            selection = self.population.index
        
        self.population.loc[selection, 'last_tested_date'] = day

        selection_antigen, selection_pcr = (
            self.get_test_type_selections(selection)
        )
        is_antigen_test = (
            self.population['id'].apply(lambda i: i in selection_antigen)
        )
        is_pcr_test = self.population['id'].apply(lambda i: i in selection_pcr)
        
        passes_sensitivity = (
            (np.random.rand(self.N) < self.sensitivity_antigen_sym) &
            self.population.is_symptomatic &
            is_antigen_test
            
        ) | (
            (np.random.rand(self.N) < self.sensitivity_antigen_asy) &
            ~self.population.is_symptomatic & 
            is_antigen_test
        ) | (
             (np.random.rand(self.N) < self.sensitivity_pcr) &
            is_pcr_test
        )
        will_be_detected = (
            self.population['id'].apply(lambda i: i in selection) &
            (self.population['state'] == 'I') &
            passes_sensitivity & 
            (
                (day - self.population['infection_date']) 
                >= self.infection_to_detectable_delay
            )
        )
        self.population.loc[
            will_be_detected, 'positive_test_dates'
        ] = self.population.loc[
            will_be_detected, 'positive_test_dates'
        ].apply(lambda s: s.union({day}))

        return is_antigen_test, is_pcr_test
             
    def receive_test_results(self, day, is_antigen_test, is_pcr_test):
        """Receive earlier test results and place new positives in quarantine.
        """
        is_detected = (
            self.population['positive_test_dates'].apply(
                lambda test_dates: (day - self.delay_antigen) in test_dates
            ) & is_antigen_test
        ) | (
            self.population['positive_test_dates'].apply(
                lambda test_dates: (day - self.delay_pcr) in test_dates
            ) & is_pcr_test
        )
        
        # If is already quarantining due to previous positive test, don't 
        # restart `quarantine_start_date` counter
        was_already_quarantining = (
            (self.population.state == 'Q') &
            (self.population.positive_test_dates.apply(
                lambda dates: len(dates) > 0)
            )
        )
        self.population.loc[is_detected, 'state'] = 'Q'
        self.population.loc[
            is_detected & ~was_already_quarantining, 'quarantine_start_date'
        ] = day
        
        self.end_self_quarantine_if_neg(day, is_antigen_test, is_pcr_test)
    
    def end_self_quarantine_if_neg(self, day, is_antigen_test, is_pcr_test):
        """End self-imposed quarantine if negative result.
        """
        is_not_detected = (
            self.population['negative_test_dates'].apply(
                lambda test_dates: (day - self.delay_antigen) in test_dates
            ) & is_antigen_test
        ) | (
            self.population['negative_test_dates'].apply(
                lambda test_dates: (day - self.delay_pcr) in test_dates
            ) & is_pcr_test
        )

        is_neg_and_quarantining = (
            (self.population.state == 'Q') & is_not_detected 
        )
        is_true_neg_and_quarantining = (
            is_neg_and_quarantining & 
            self.population.infection_date.apply(lambda x: math.isnan(x))
        )
        is_false_neg_and_quarantining = (
            is_neg_and_quarantining & 
            self.population.infection_date.apply(lambda x: ~math.isnan(x))
        )
        self.population.loc[
            is_neg_and_quarantining, 'quarantine_start_date'
        ] = None
        self.population.loc[is_true_neg_and_quarantining, 'state'] = 'S'
        self.population.loc[is_false_neg_and_quarantining, 'state'] = 'I'

    def release_quarantined_cases(self, day):
        """Release anyone who has finished quarantine.
        """
        have_quarantined_full_duration = (
            (day - self.population['quarantine_start_date']) == self.Q_duration
        )
        
        # Assume symptoms vanish after quarantine period
        self.population.loc[
            have_quarantined_full_duration, 'is_symptomatic'
        ] = False
        
        to_release_recovered = (
            have_quarantined_full_duration & 
            self.population['infection_date'].apply(lambda d: ~math.isnan(d))
        )
        to_release_susceptible = (
            have_quarantined_full_duration & 
            self.population['infection_date'].apply(lambda d: math.isnan(d))
        )
                
        self.population.loc[to_release_recovered, 'state'] = 'R'
        self.population.loc[to_release_susceptible, 'state'] = 'S'
    
    def recover_infected_cases(self):
        """Infected --> Recovered transition.
        """
        self.population.loc[
            (
                (self.population['state'] == 'I') & 
                (np.random.rand(self.N) < self.gamma)
            ), 'state'
        ] = 'R'
    
    def introduce_symptoms(self):
        """Certain percentage of people will develop symptoms.
        
        Percentage of those with symptoms (whether due to COVID-19 or 
        otherwise), will choose to remain at home. Update `self.population`to 
        reflect this.
        """
        # Percentage of infected show symptoms
        self.population.loc[
            (self.population['state'] == 'I') & 
            (np.random.rand(self.N) < self.sym_pos_rate), 'is_symptomatic'
        ] = True
        
        # Percentage of non-infected show symptoms
        self.population.loc[
            (self.population['state'] != 'I') & 
            (np.random.rand(self.N) < self.sym_neg_rate), 'is_symptomatic'
        ] = True
        
    def track_self_imposed_quarantine(self, day): 
        """Track cases who choose to self-isolate with onset of symptoms.

        These include a percentage (dependent on the `risk_behavior` parameter) 
        of cases who:
            -- Are not already quarantining.
            -- Are symptomatic.
            -- Have not already knowingly recovered from COVID-19.
        """
        will_self_quarantine = (
            (self.population.state != 'Q') &
            (self.population.is_symptomatic) &
            (np.random.rand(self.N) < self.risk_behavior) &  
            ~self.population.known_to_be_recovered
        )
        
        # Those showing symptoms who choose to self-quarantine
        self.population.loc[will_self_quarantine, 'state'] = 'Q'
        self.population.loc[
            will_self_quarantine, 'quarantine_start_date'
        ] = day
          
    def infect_susceptible_cases(self, day):
        """Susceptible --> Infected transition.
        
        Easiest to think of beta as the number of contact a potential recipient 
        person makes. The probability that all of those contacts are with safe 
        people is: (((N-Q)-I)/(N-Q))**beta  
        """     
        # If self.N - self.Q == 0 means everyone is quarantining
        if self.N - self.Q > 0:
            has_contact_with_internal_infected = (
                np.random.rand(self.N) < (
                    1 - (
                        (self.N - self.Q - self.I) / (self.N - self.Q)
                    )**self.beta
                )
            )
            has_contact_with_external_infected = (
                np.random.rand(self.N) < self.external_infection_rate
            )
            has_contact = (
                has_contact_with_internal_infected | 
                has_contact_with_external_infected
            )
            is_susceptible = self.population['state'] == 'S'
            is_infected = is_susceptible & has_contact
            self.population.loc[is_infected, 'state'] = 'I'
            self.population.loc[is_infected, 'infection_date'] = day
    
    def track_known_recovered_cases(self):
        """Track cases that have knowingly recovered.

        If both negative and positive test result present, and negative result
        came post positive result, track that they knowingly recovered (note 
        that case may be tracked as True due to false negative). 
        """
        has_both_pos_and_neg = self.population.negative_test_dates.apply(
            lambda x: len(x) > 0
        ) & self.population.positive_test_dates.apply(
            lambda x: len(x) > 0
        )

        self.population.loc[has_both_pos_and_neg, 'known_to_be_recovered'] = (
            self.population.loc[has_both_pos_and_neg, :].apply(
                lambda x: (
                    max(x['negative_test_dates']) > 
                    max(x['positive_test_dates'])
                ), axis=1,
            )  
        )

    def run_simulation(self):
        """Run COVID-19 simulation.

        Introduce initial infections --> Introduce symptoms --> Track 
        self-imposed quarantine --> Release quarantined cases --> Run tests -->
        Receive test results --> Recover infected cases --> Infect susceptible
        cases.
        """
        self.introduce_initial_infections()
        self.introduce_initial_recovered()

        for day in range(self.num_days):
            self.log_states(day)
            
            self.introduce_symptoms()
            self.track_known_recovered_cases()
            self.track_self_imposed_quarantine(day)
            self.release_quarantined_cases(day)

            if self.testing_interval is not None:
                # if a testing day, run_tests:
                if day % self.testing_interval == 0:
                    is_antigen_test, is_pcr_test = self.run_tests(day)
                
                self.receive_test_results(day, is_antigen_test, is_pcr_test)
            
            self.recover_infected_cases()
            self.infect_susceptible_cases(day)
            self.log_updated_states(day)
        
        return (
            pd.DataFrame(self.state_counts).T.fillna(0), 
            self.cumulative_infections, 
            self.population, 
            pd.concat(self.state_logs, axis=1),
        )
