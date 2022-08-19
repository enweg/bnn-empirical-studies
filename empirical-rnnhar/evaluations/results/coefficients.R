library(tidyverse)
library(data.table)
rm(list = ls())

path <- "../"
files <- list.files(path)
files <- files[str_detect(files, "coeffs")]

files_ggmc <- files[!str_detect(files, "bbb")]
files_ggmc <- paste0(path, files_ggmc)
files_bbb <- files[str_detect(files, "bbb")]
files_bbb <- paste0(path, files_bbb)

read_data <- function(f){
  netid <- str_match(f, "net([0-9]*)")[,2]
  rep <- str_match(f, "rep([0-9]*)")[,2]
  tmp <- fread(f)
  tmp[, netid := netid]
  tmp[, rep := rep]
  tmp[, time := 1:nrow(tmp)]
  return(tmp)
}

# Main findings
# - Runs of the same network result in very different coefficient paths
# - This is not related to the mixing in sigma or rmse performance at
# either log or level, nor to qqdiff performance
# - when coefficient paths are standardised, they look much more similar
# and two maybe three goups of coefficient path types seem to exist.
ggmc <- do.call(rbind, lapply(files_ggmc, read_data))
ggmc <- as_tibble(ggmc)

# Changing ggmc and bbb in the graphs below givest he following results
# - bbb estimates much smaller time variation than ggmc; It also estimates
# very different time variation even if the same network is used
# - The fact that LSTM intercept are higher than RNN no-longer holds
# - time variation still seems to have nothing to do with rmse
# - Time variation looks to have two maybe three goups when standardised
bbb <- do.call(rbind, lapply(files_bbb, read_data))
bbb <- as_tibble(bbb)

# Even runs of the same network learn largely varying coefficient paths
# this might be due to the multimodality
# how does this relate to performance?
# Fact holds for all variables
ggmc %>%
  filter(netid == 4) %>%
  mutate(group = paste0(netid, "/", rep)) %>%
  ggplot(aes(x = time, y = intercept, group = group)) + 
  geom_line(alpha = 0.1)

# If intercept is very noisy, all others will be very noisy
# too or will have very little variation
ggmc %>%
  filter(netid == 1 & rep == 10) %>%
  mutate(group = paste0(netid, "/", rep)) %>%
  ggplot(aes(x = time, y = intercept, group = group)) + 
  geom_line(alpha = 1)
  

# Taking averages across all runs better reveals the time variation. 
# LSTM networks seem to estimate the intercept to be higher than RNN networks
ggmc %>%
  group_by(netid, time) %>%
  summarise(across(.cols = c(intercept, daily, weekly, monthly), 
                   .fns = mean), 
            .groups = "drop") %>%
  mutate(kind = ifelse(netid <= 2, "RNN", "LSTM")) %>%
  ggplot(aes(x = time, y = intercept, group = netid, color = kind)) + 
  geom_line()


files <- list.files(path)
files <- files[str_detect(files, "stats")]
files_ggmc <- files[!str_detect(files, "bbb|baseline")]
files_bbb <- files[str_detect(files, "bbb")]
ggmc_stats <- do.call(rbind, lapply(paste0(path, files_ggmc), fread))
ggmc_stats <- as_tibble(ggmc_stats)
bbb_stats <- do.call(rbind, lapply(paste0(path, files_bbb), fread))
bbb_stats <- as_tibble(bbb_stats)
num_take <- 10

# How much coefficient vary seems to have no relationship to mixing
best_mixing <- ggmc_stats %>% 
  select(net, rep, rhat_sigma) %>%
  arrange(rhat_sigma) %>%
  head(num_take) %>%
  mutate(mixing = "good") %>%
  rename(netid = net)
worst_mixing <- ggmc_stats %>% 
  select(net, rep, rhat_sigma) %>%
  arrange(rhat_sigma) %>%
  tail(num_take) %>%
  mutate(mixing = "bad") %>%
  rename(netid = net)
best_mixing %>%
  rbind(worst_mixing) %>%
  mutate(netid = as.character(netid), 
         rep = as.character(rep), 
         group = paste0(netid, "/", rep)) %>%
  inner_join(ggmc, by = c("netid", "rep")) %>%
  ggplot(aes(x = time, y = intercept, group = group)) + 
  geom_line() + 
  facet_wrap(vars(mixing))

# How much coefficients vary seems to have nothing to do with RMSE
# Not in logs nor in levels
best <- ggmc_stats %>% 
  select(net, rep, rmse_level) %>%
  arrange(rmse_level) %>%
  head(num_take) %>%
  mutate(stat = "good") %>%
  rename(netid = net)
worst <- ggmc_stats %>% 
  select(net, rep, rmse_level) %>%
  arrange(rmse_level) %>%
  tail(num_take) %>%
  mutate(stat = "bad") %>%
  rename(netid = net)
best %>%
  rbind(worst) %>%
  mutate(netid = as.character(netid), 
         rep = as.character(rep), 
         group = paste0(netid, "/", rep)) %>%
  inner_join(ggmc, by = c("netid", "rep")) %>%
  ggplot(aes(x = time, y = intercept, group = group)) + 
  geom_line() + 
  facet_wrap(vars(stat))


# No relationship to qqdiff performance
# files <- list.files(path)
# files <- files[str_detect(files, "qperformance")]
# files_ggmc <- files[!str_detect(files, "bbb")]
# ggmc_qperformance <- do.call(rbind, lapply(paste0(path, files_ggmc), read_data))
# ggmc_qperformance <- as_tibble(ggmc_qperformance)
# ggmc_qperformance <- ggmc_qperformance %>%
#   group_by(netid, rep) %>%
#   summarise(qqdiff = sum(abs(observed - target)), 
#             .groups = "drop")
# 
# ggmc_qperformance %>%
#   arrange(qqdiff)
# 
# ggmc %>%
#   filter(netid == 2 & rep == 6) %>%
#   ggplot(aes(x = time, y = intercept)) + 
#   geom_line()

# There always seem to be two, maybe three groups of time variation
# This only applies when time variation is standardised
ggmc %>%
  filter(netid == 1) %>%
  mutate(group = paste0(netid, "/", rep)) %>%
  group_by(netid, rep) %>%
  mutate(across(.cols = c(intercept, daily, weekly, monthly),
                .fns = function(x) (x - mean(x))/sd(x))) %>%
  ungroup() %>%
  ggplot(aes(x = time, y = monthly)) + 
  geom_line(alpha = 1) +
  facet_wrap(vars(group))

