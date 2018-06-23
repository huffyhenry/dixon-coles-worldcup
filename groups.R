# Quick & dirty Dixon-Coles predictions for the group stage. Lifted wholesale from
# http://www.statsandsnakeoil.com/2018/06/05/modelling-the-world-cup-with-regista/.
# Depends on the regista package available from https://github.com/Torvaney/regista.

library(dplyr)
library(stringr)
library(regista)
library(tidyr)

data <- read.csv("results.csv") %>%
  mutate(hfa=ifelse(neutral, 0, 1),
         date=as.Date(date)) %>%
  filter(date > '2016-06-01')

teams <-   bind_rows(data %>% select(team = home_team),
                     data %>% select(team = away_team)) %>%
  count(team) %>%
  filter(n >= 20) %>%
  .$team

filtered_games <-
  data %>%
  filter(home_team %in% teams,
         away_team %in% teams) %>%
  # We can use a regista function to convert teamnames to factors
  # that share the same levels:
  factor_teams(c("home_team", "away_team"))

str_glue("Number of teams: {length(teams)}
          Number of games: {nrow(filtered_games)}")

res <- dixoncoles(~home_score, ~away_score, ~home_team, ~away_team, filtered_games)

estimates <-
  tibble(parameter = names(res$par),
         value     = res$par) %>%
  separate(parameter, c("parameter", "team"), "___") %>%
  mutate(value = exp(value))

.dc.pmf <- function(x, y, alpha1, beta1, alpha2, beta2, rho){
  # The probability mass function of the Dixon-Coles distribution w/o HFA
  # Consult the original paper for the definition.
  lambda <- alpha1*beta2  # No HFA
  mu <- alpha2*beta1
  if ((x == 0) && (y == 0))
    tau <- 1 - lambda*mu*rho
  else if ((x == 0) && (y == 1))
    tau <- 1 + lambda * rho
  else if ((x == 1) && (y == 0))
    tau <- 1 + mu*rho
  else if ((x == 1) && (y == 1))
    tau <- 1 - rho
  else
    tau <- 1

  return(tau * lambda^x * exp(-lambda) * mu^y * exp(-mu) / factorial(x) / factorial(y))
}

predict <- function(team1, team2, params){
  # Print probabilites of all scores from 0:0 to 3:3
  max.goals <- 3
  alpha1 <- params %>% filter(team == team1) %>% filter(parameter == "off") %>% pull(value)
  alpha2 <- params %>% filter(team == team2) %>% filter(parameter == "off") %>% pull(value)
  beta1 <- params %>% filter(team == team1) %>% filter(parameter == "def") %>% pull(value)
  beta2 <- params %>% filter(team == team2) %>% filter(parameter == "def") %>% pull(value)
  rho <- params %>% filter(parameter == "rho") %>% pull(value)
  print(sprintf("Predicting %s - %s:", team1, team2))
  for (i in seq(0, max.goals)){
    for (j in seq(0, max.goals)){
      prob <- .dc.pmf(i, j, alpha1, beta1, alpha2, beta2, log(rho))
      print(sprintf("%d - %d: %.1f%%", i, j, 100*prob))
    }
  }
}
