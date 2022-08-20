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
ggmc <- do.call(rbind, lapply(files_ggmc, read_data))
ggmc <- as_tibble(ggmc)

# - runs of same network result in very different coefficient paths
# - Above is not related to performance
# - standardised paths look more similar but not as similar as for GGMC
bbb <- do.call(rbind, lapply(files_bbb, read_data))
bbb <- as_tibble(bbb)

# Even runs of the same network learn largely varying coefficient paths
# this might be due to the multimodality
# how does this relate to performance?
# Fact holds for all variables
p <- ggmc %>%
  filter(netid == 2) %>%
  mutate(group = paste0(netid, "/", rep)) %>%
  ggplot(aes(x = time, y = intercept, color = group)) + 
  geom_line(alpha = 1) + 
  xlab("") + ylab("Intercept Coefficient") +
  theme_bw() + 
  theme(legend.position = "none")
p
ggsave("./varying-coefficients.pdf", plot = p, device = "pdf", width = 15, height = 7)


files <- list.files(path)
files <- files[str_detect(files, "stats")]
files_ggmc <- files[!str_detect(files, "bbb|baseline")]
files_bbb <- files[str_detect(files, "bbb")]
ggmc_stats <- do.call(rbind, lapply(paste0(path, files_ggmc), fread))
ggmc_stats <- as_tibble(ggmc_stats)
bbb_stats <- do.call(rbind, lapply(paste0(path, files_bbb), fread))
bbb_stats <- as_tibble(bbb_stats)
num_take <- 2

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



# There always seem to be two, maybe three groups of time variation
# This only applies when time variation is standardised
ggmc %>%
  filter(netid == 2) %>%
  mutate(group = paste0(netid, "/", rep)) %>%
  group_by(netid, rep) %>%
  mutate(across(.cols = c(intercept, daily, weekly, monthly),
                .fns = function(x) (x - mean(x))/sd(x))) %>%
  ungroup() %>%
  ggplot(aes(x = time, y = intercept)) + 
  geom_line(alpha = 1) +
  facet_wrap(vars(group))


# Not sure how to interpret the correlations
get_correlations <- function(group){
  cormat <- cor(group[,1:4])
  name <- c()
  cnames <- colnames(cormat)
  for (i in 2:ncol(cormat)){
    name <- c(name, paste0(cnames[1:i-1], "_", cnames[i]))   
  }
  cormat <- cormat[upper.tri(cormat)]
  names(cormat) <- name
  return(data.frame(cor = cormat, relation = names(cormat), row.names = NULL))
}

# library(plyr)
# variables <- c("intercept", "daily", "weekly", "monthly")
# coeff_cors <- ddply(select(ggmc, intercept, daily, weekly, monthly, everything()), .(netid, rep), get_correlations)
# coeff_cors %>%
#  as_tibble() %>%
#   mutate(netid = as.character(netid), 
#          var1 = str_split_fixed(relation, "_", n = Inf)[, 1], 
#          var2 = str_split_fixed(relation, "_", n = Inf)[, 2]) %>%
#   complete(var1 = variables, 
#            var2 = variables) %>%
#   mutate(var1f = factor(var1, levels = variables), 
#          var2f = factor(var2, levels = variables)) %>%
#   ggplot() + 
#   geom_boxplot(aes(x = netid, y = cor)) + 
#   facet_grid(rows = vars(var1f), cols = vars(var2f), space = "free") + 
#   theme_bw()

