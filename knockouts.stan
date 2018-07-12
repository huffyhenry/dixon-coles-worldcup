functions {
  real tau(int[] goals, real[] mu, real rho){
    // The adjustment factor for the probability of low-scoring games.
    // Cf. the original Dixon-Coles paper.
    if ((goals[1] == 0) && (goals[2] == 0))
      return 1.0 - rho*mu[1]*mu[2];
    else if ((goals[1] == 1) && (goals[2] == 0))
      return 1.0 + rho*mu[2];
    else if ((goals[1] == 0) && (goals[2] == 1))
      return 1.0 + rho*mu[1];
    else if ((goals[1] == 1) && (goals[2] == 1))
      return 1.0 - rho;
    else
      return 1.0;
  }

  real dixoncoles_log(int[] goals, real[] mu, real rho){
    return poisson_lpmf(goals[1] | mu[1]) +
           poisson_lpmf(goals[2] | mu[2]) +
           log(tau(goals, mu, rho));

  }
}

data {
  int<lower=2> n_teams;
  int<lower=1> n_games;

  int<lower=1, upper=n_teams> team1[n_games];
  int<lower=1, upper=n_teams> team2[n_games];
  int<lower=0> goals1[n_games];
  int<lower=0> goals2[n_games];
  int<lower=0, upper=1> neutral[n_games];
  int<lower=0, upper=1> worldcup[n_games];
}

parameters {
  // Offense, defence and HFA are in log space
  real off_raw[n_teams-1];
  real def_raw[n_teams-1];
  real hfa;
  real rho;
}

transformed parameters {
  // Introduce sum-to-zero constraint on defence and offence ratings
  real off[n_teams];
  real def[n_teams];

  for (i in 1:n_teams - 1){
    off[i] = off_raw[i];
    def[i] = def_raw[i];
  }

  off[n_teams] = -sum(off_raw);
  def[n_teams] = -sum(def_raw);
}

model {
  int score[2];
  real mu[2];

  off ~ normal(0, 1);
  def ~ normal(0, 1);
  hfa ~ normal(0.25, 0.1);
  rho ~ normal(0, 0.1);

  for (i in 1:n_games){
    score[1] = goals1[i];
    score[2] = goals2[i];
    mu[1] = exp(off[team1[i]] + def[team2[i]] + (1 - neutral[i])*hfa);
    mu[2] = exp(off[team2[i]] + def[team1[i]]);

    for (j in 1:(2*worldcup[i] + 1)){  // Count World Cup games thrice
      score ~ dixoncoles(mu, rho);
    }
  }
}
