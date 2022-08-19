library(tidyverse)
library(data.table)
rm(list = ls())

path <- "../"
files <- list.files(path)
files <- files[str_detect(files, "stats")]
files_ggmc <- files[!str_detect(files, "bbb|baseline")]
files_bbb <- files[str_detect(files, "bbb")]
files_baseline <- files[str_detect(files, "baseline")]
ggmc_stats <- do.call(rbind, lapply(paste0(path, files_ggmc), fread))
ggmc_stats <- as_tibble(ggmc_stats) %>% mutate(method = "GGMC")
bbb_stats <- do.call(rbind, lapply(paste0(path, files_bbb), fread))
bbb_stats <- as_tibble(bbb_stats) %>% mutate(method = "BBB")
baseline_stats <- read_csv(paste0(path, files_baseline))
baseline_rmse <- baseline_stats %>%
  select(starts_with("rmse")) %>%
  pivot_longer(everything(), 
               names_to = c(".value", "transform"), 
               names_sep = "_") %>%
  mutate(transform = ifelse(transform == "level", "RV", "log(RV)")) %>%
  rename(har_rmse = rmse)
baseline_var <- baseline_stats %>%
  select(starts_with("VaR")) %>%
  pivot_longer(everything(), 
               names_to = "measure", 
               values_to = "har_var")


# We never managed to achieve good mixing
ggmc_stats %>%
  mutate(net = as.character(net)) %>%
  select(net, rhat_sigma, starts_with("rmse")) %>%
  pivot_longer(starts_with("rmse"), 
               names_to = c(".value", "transform"), 
               names_sep = "_") %>%
  mutate(transform = ifelse(transform == "level", "RV", "log(RV)"), 
         good_mixing = rhat_sigma < 1.01) %>%
  left_join(baseline_rmse, by = "transform") %>%
  ggplot(aes(x = good_mixing, y = rmse)) + 
  geom_boxplot() + 
  geom_hline(aes(yintercept = har_rmse, color = "HAR Benchmark")) + 
  facet_grid(cols = vars(net), rows = vars(transform), scales = "free") + 
  scale_color_manual(values = c("HAR Benchmark" = "red")) +
  guides(color = guide_legend(title = "")) +
  xlab("Good Mixing") + 
  ylab("RMSE") +
  theme_bw() + 
  theme(legend.position = "top")

stats <- rbind(select(ggmc_stats, -rhat_sigma), bbb_stats)

# GGMC performs clearly better for log RV
# Performance seems to be almost the same for RV in levels 
# GGMC is much more consistent with performance though, shown by narrower boxes
stats %>%
  mutate(net = as.character(net)) %>%
  select(net, method, starts_with("rmse")) %>%
  pivot_longer(starts_with("rmse"), 
               names_to = c(".value", "transform"), 
               names_sep = "_") %>%
  mutate(transform = ifelse(transform == "level", "RV", "log(RV)")) %>%
  left_join(baseline_rmse, by = "transform") %>%
  ggplot(aes(x = net, y = rmse)) + 
  geom_boxplot() + 
  geom_hline(aes(yintercept = har_rmse, color = "HAR Benchmark")) + 
  facet_grid(cols = vars(method), rows = vars(transform), scales = "free") + 
  scale_color_manual(values = c("HAR Benchmark" = "red")) +
  guides(color = guide_legend(title = "")) +
  xlab("Network ID") + 
  ylab("RMSE") +
  theme_bw() + 
  theme(legend.position = "top")

var_translate = c(
  "VaR_0_1" = "VaR 0.1%", 
  "VaR_1_0" = "VaR 1%", 
  "VaR_5_0" = "VaR 5%", 
  "VaR_10_0" = "VaR 10%"
)

var_target = c(
  "VaR_0_1" = 0.001, 
  "VaR_1_0" = 0.01, 
  "VaR_5_0" = 0.05, 
  "VaR_10_0" = 0.1
)

var_ordering <- c("VaR 0.1%", "VaR 1%", "VaR 5%", "VaR 10%")

# Both not good for VaR
# BBB performance sensitive to network while GGMC performance 
# is consistent
stats %>%
  mutate(net = as.character(net)) %>%
  select(net, method, starts_with("VaR")) %>%
  pivot_longer(starts_with("VaR"), 
               names_to = "measure", 
               values_to = "values") %>%
  left_join(baseline_var, by = "measure") %>%
  mutate(target = var_target[measure], 
         measure = var_translate[measure], 
         measure = factor(measure, levels = var_ordering)) %>%
  ggplot(aes(x = net, y = values)) + 
  geom_boxplot() + 
  geom_hline(aes(yintercept = target, color = "Target"), linetype = "dotted", size = 1) + 
  geom_hline(aes(yintercept = har_var, color = "HAR Benchmark"), size = 1) +
  facet_grid(cols = vars(method), rows = vars(measure), scales = "free") + 
  scale_color_manual(values = c("Target" = "black", "HAR Benchmark" = "red")) + 
  scale_y_continuous(labels = scales::percent) + 
  guides(color = guide_legend(title = "")) + 
  xlab("Network ID") + ylab("% test data falling below VaR") +
  theme_bw() + 
  theme(legend.position = "top")
