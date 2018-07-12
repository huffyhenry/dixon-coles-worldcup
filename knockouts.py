""" Modelling the knockout stage of the 2018 World Cup. """
import os
import pickle
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
import pandas as pd
import pystan


def wrangle():
    data = pd.read_csv("results.csv", index_col=None, parse_dates=['date'])
    team_map = {
        name: idx + 1
        for idx, name in enumerate(set(data.home_team) | set(data.away_team))
    }
    stan_data = (
        data
        [data.date > dt.datetime.today() - dt.timedelta(days=2*365)]
        .assign(team1=lambda df: list(map(team_map.__getitem__, df.home_team)),
                team2=lambda df: list(map(team_map.__getitem__, df.away_team)),
                goals1=lambda df: df.home_score,
                goals2=lambda df: df.away_score)
        [['team1', 'team2', 'goals1', 'goals2', 'worldcup', 'neutral']]
        .to_dict(orient='list')
    )
    stan_data['n_teams'] = len(team_map)
    stan_data['n_games'] = len(stan_data['team1'])

    return stan_data, team_map


def fit(data, force_compile=False, **kwargs):
    if not force_compile and os.path.exists("cache/knockouts.pickle"):
        model = pickle.load(open("cache/knockouts.pickle", 'rb'))
    else:
        model = pystan.StanModel("knockouts.stan")
        pickle.dump(model, open("cache/knockouts.pickle", 'wb'))

    # Starting the chains from 0, as they struggle otherwise.
    # The model is in log space, making 0 a convenient starting point anyway.
    return model.sampling(data, init=0, **kwargs)


def _dixoncoles_pmf(goals, mus, rho):
    """ The probability mass function of the Dixon-Coles distribution. """
    def tau():
        if goals[0] == goals[1] == 0:
            return 1.0 - rho*mus[0]*mus[1]
        elif goals[0] == 0 and goals[1] == 1:
            return 1.0 + rho*mus[0]
        elif goals[0] == 1 and goals[1] == 0:
            return 1.0 + rho*mus[1]
        elif goals[0] == goals[1] == 1:
            return 1.0 - rho
        else:
            return 1.0

    return tau() * poisson.pmf(goals[0], mus[0]) * poisson.pmf(goals[1], mus[1])


def graph(samples, team_map):
    """ Plot model fit for quick inspection. """
    fig = plt.figure()
    ax = plt.subplot(111)
    for name, idx in team_map.items():
        off_ = np.mean(np.exp(samples['off'][:, idx-1]))
        def_ = np.mean(np.exp(samples['def'][:, idx-1]))
        ax.scatter([off_], [def_])
        ax.annotate(name, (off_, def_))
    ax.set_xlabel("Attack")
    ax.set_ylabel("Defence")
    plt.show()


def predict(samples, team_map, team1, team2, neutral=True, aet=True, max_goals=4):
    """ Display the prediction for an arbitrary match-up. """
    idx1 = team_map[team1] - 1
    idx2 = team_map[team2] - 1

    off1 = np.mean(np.exp(samples['off'][:, idx1]))
    off2 = np.mean(np.exp(samples['off'][:, idx2]))
    def1 = np.mean(np.exp(samples['def'][:, idx1]))
    def2 = np.mean(np.exp(samples['def'][:, idx2]))
    hfa = np.mean(np.exp(samples['hfa']))
    rho = np.mean(samples['rho'])

    mus = (off1 * def2 * (1.0 if neutral else hfa), off2 * def1)

    probs = np.zeros((max_goals + 1, max_goals + 1))

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            probs[h, a] = _dixoncoles_pmf((h, a), mus, rho)

    if aet:
        # Compute probabilities of scoring in extra time by taking
        # the DC pmf with the scoring rates divided by 3.
        extra = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                extra[h, a] = _dixoncoles_pmf((h, a), (0.33*mus[0], 0.33*mus[1]), rho)

        # Copy the probabilities of draws in 90 mins and zero the entries
        base = [probs[g, g] for g in range(max_goals + 1)]
        for g in range(max_goals + 1):
            probs[g, g] = 0.0

        # Redistribute the original probabilities of draws
        for g in range(max_goals + 1):
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    if (g + h <= max_goals) and (g + a) <= max_goals:
                        probs[g + h, g + a] += base[g] * extra[h, a]

    fig = plt.figure()
    ax = plt.subplot(111)
    img = ax.imshow(probs.T, origin='upper')
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            plt.text(h, a, "%.1f%%" % (100.0 * probs[h, a]),
                     horizontalalignment='center',
                     verticalalignment='center')
            print("%d-%d: %.1f%%" % (h, a, 100.0 * probs[h, a]))
    ax.set_title("FT/AET score prediction")
    ax.set_xlabel(team1)
    ax.set_ylabel(team2)
    plt.colorbar(img)
    plt.show()
