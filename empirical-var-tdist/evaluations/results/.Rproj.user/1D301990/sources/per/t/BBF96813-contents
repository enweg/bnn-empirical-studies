library(tidyverse)

benchmark <- read_csv("../tdist-garch11-baseline.csv")
single_chain <- read_csv("../mcmc-single-chain-evaluation-all.csv")
same_network <- read_csv("../mcmc-same-network-evaluation-all.csv")
bbb <- read_csv("../bbb-single-asset-all.csv")

benchmark_long <- benchmark %>%
  pivot_longer(-nu, 
               names_to = "measure", 
               values_to = "garch")

bbb_long <- bbb %>%
  pivot_longer(-c(network, nu), 
               names_to = "measure", 
               values_to = "bbb")

bbb_nu <- bbb_long %>%
  group_by(nu, measure) %>%
  summarise(bbb = mean(bbb), .groups = "drop")

bbb_network <- bbb_long %>%
  group_by(network, measure) %>%
  summarise(bbb = mean(bbb), .groups = "drop") %>%
  mutate(network = factor(network, levels = sort(unique(network))))

single_chain_comparison <- single_chain %>%
  select(-c(mape, config, network)) %>%
  mutate(well_mixing = rhat_sigma <= 1.01) %>%
  pivot_longer(-c(well_mixing, nu), 
               names_to = "measure", 
               values_to = "values") %>%
  filter(measure != "rhat_sigma") %>%
  left_join(benchmark_long, by = c("measure", "nu"))

same_network_evaluation <- same_network %>%
  pivot_longer(-network, 
               names_to = "measure", 
               values_to = "values") %>%
  left_join(benchmark_long, by = c("measure")) %>%
  filter(!measure %in% c("rhat_sigma", "rmse", "mape")) %>%
  mutate(network = factor(network, levels = sort(unique(network))))

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

p_single_chain <- single_chain_comparison %>%
  left_join(bbb_nu, by = c("nu", "measure")) %>%
  filter(measure != "rmse") %>%
  mutate(target = var_target[measure]) %>%
  mutate(measure = ifelse(measure %in% names(var_translate), 
                          var_translate[measure], 
                          measure)) %>%
  ggplot(aes(y = values, x = well_mixing)) + 
  geom_hline(aes(yintercept = garch, color = "Garch(1, 1)"), size = 1) + 
  geom_hline(aes(yintercept = target, color = "Target"), size = 1, linetype = "dotted") + 
  geom_hline(aes(yintercept = bbb, color = "BBB"), size = 1) +
  geom_boxplot() + 
  facet_grid(rows = vars(measure), cols = vars(nu), scales = "free") + 
  scale_color_manual(values = c("Garch(1, 1)" = "red", "Target" = "black", "BBB" = "blue")) + 
  scale_y_continuous(labels = scales::percent) + 
  guides(color = guide_legend(title = "")) + 
  xlab("Good mixing in sigma?") + ylab("% test data falling below VaR") + 
  theme_bw() + 
  theme(legend.position = "top")

p_single_chain
ggsave("./single-asset-single-chain-tgarch.pdf", 
       plot = p_single_chain, 
       device = "pdf")

single_chain_comparison %>%
  filter(measure == "rmse") %>%
  group_by(nu) %>%
  summarise(min = min(values), 
            max = max(values), 
            mean = mean(values)) %>%
  write_csv(., file = "./single-asset-nu-rmse-tgarch.csv")

p_same_network <- same_network_evaluation %>%
  left_join(bbb_network, by = c("network", "measure")) %>%
  mutate(target = var_target[measure]) %>%
  mutate(measure = var_translate[measure]) %>%
  mutate(kind = ifelse(as.numeric(network) <= 2, "RNN", "LSTM")) %>%
  ggplot(aes(x = network, y = values)) + 
  geom_col(aes(fill = kind),position = "dodge2", width = 0.7, alpha = 0.5) + 
  geom_hline(aes(yintercept = garch, color = "Garch(1, 1)", group = nu), size = 0.5) + 
  geom_hline(aes(yintercept = target, color = "Target"), size = 1, linetype = "dotted") + 
  geom_point(aes(y = bbb, color = "BBB"), shape = 4) +
  facet_wrap(vars(measure), scales = "free") + 
  scale_color_manual(values = c("Garch(1, 1)" = "red", "Target" = "black", "BBB" = "blue")) + 
  scale_y_continuous(labels = scales::percent) + 
  guides(color = guide_legend(title = ""), 
         fill = guide_legend(title = "Network Type")) + 
  xlab("Network ID") + ylab("% test data falling below VaR") +
  theme_bw() + 
  theme(legend.position = "top")

p_same_network
ggsave("./single-asset-same-network-tgarch.pdf", 
       plot = p_same_network, 
       device = "pdf")


p_mixing_performance <- single_chain %>%
  mutate(kind = ifelse(network <= 2, "RNN", "LSTM")) %>%
  select(rhat_sigma, starts_with("VaR"), kind) %>%
  pivot_longer(-c(rhat_sigma, kind), 
               names_to = "measure", 
               values_to = "values") %>%
  mutate(target = var_target[measure], 
         measure = var_translate[measure]) %>%
  ggplot(aes(x = rhat_sigma, y = values)) + 
  geom_point(aes(color = kind)) + 
  facet_wrap(vars(measure), scales = "free_y") +
  scale_y_continuous(labels = scales::percent) + 
  xlab("Rhat sigma") + ylab("% test data falling below VaR") +
  guides(color = guide_legend(title = "Network Type")) +
  theme_bw() + 
  theme(legend.position = "top")

p_mixing_performance
ggsave("./single-asset-mixing-performance.pdf", 
       plot = p_mixing_performance, 
       device = "pdf")
