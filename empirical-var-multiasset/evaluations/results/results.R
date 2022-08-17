library(data.table)
library(tidyverse)
rm(list = ls())

path <- "../"
files <- list.files(path)
files <- files[str_detect(files, "mcmc-single-chain-config")]

ggmc <- do.call(rbind, lapply(paste0(path, files), fread))
ggmc <- as_tibble(ggmc)
ggmc_long <- ggmc %>%
  select(-config) %>%
  pivot_longer(cols = c(rmse, starts_with("VaR")), 
               names_to = "measure", 
               values_to = "GGMC")

baseline <- read_csv("../garch11-baseline-multiple.csv")
baseline <- baseline %>%
  select(-rmse) %>%
  pivot_longer(everything(), 
               names_to = "measure", 
               values_to = "garch11")

files <- list.files(path)
files <- files[str_detect(files, "bbb-single-chain-config")]
bbb <- do.call(function(...) rbind(..., fill = TRUE), lapply(paste0(path, files), fread))
bbb <- as_tibble(bbb)
bbb_long <- bbb %>%
  select(-config) %>%
  pivot_longer(cols = c(rmse, starts_with("VaR")), 
               names_to = "measure", 
               values_to = "BBB")
  
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

all <- ggmc_long %>%
  left_join(bbb_long, by = c("netid", "m0", "measure")) %>%
  left_join(baseline, by = "measure")

p <- all %>%
  pivot_longer(cols = c(GGMC, BBB), 
               names_to = "method", 
               values_to = "values") %>%
  filter(measure != "rmse") %>%
  mutate(netid = as.character(netid), 
         target = var_target[measure], 
         measure = var_translate[measure],
         measure = factor(measure, levels = var_ordering), 
         kind = ifelse(netid <= 2, "RNN", "LSTM")) %>%
  ggplot() + 
  geom_boxplot(aes(x = netid, y = values, fill = kind), alpha = 0.5) + 
  geom_hline(aes(yintercept = garch11, color = "Garch(1, 1)"), size = 1) + 
  geom_hline(aes(yintercept = target, color = "Target"), linetype = "dotted", size = 1) + 
  facet_grid(rows = vars(method), cols = vars(measure), scales = "free") + 
  scale_color_manual(values = c("Target" = "black", "Garch(1, 1)" = "red")) +
  scale_y_continuous(labels = scales::percent) + 
  guides(color = guide_legend(title = ""), 
         fill = guide_legend(title = "Network Type")) + 
  xlab("Network ID") + ylab("% test data falling below VaR") +
  theme_bw() + 
  theme(legend.position = "top")
p
ggsave("./same-network.pdf", plot = p, device = "pdf", width = 15, height = 7)



p <- all %>%
  pivot_longer(cols = c(GGMC, BBB), 
               names_to = "method", 
               values_to = "values") %>%
  filter(measure != "rmse") %>%
  mutate(netid = as.character(netid), 
         m0 = factor(as.character(m0), levels = as.character(sort(unique(m0)))),
         target = var_target[measure], 
         measure = var_translate[measure],
         measure = factor(measure, levels = var_ordering), 
         kind = ifelse(netid <= 2, "RNN", "LSTM")) %>%
  ggplot() + 
  geom_boxplot(aes(x = m0, y = values, fill = kind), alpha = 0.5) + 
  geom_hline(aes(yintercept = garch11, color = "Garch(1, 1)"), size = 1) + 
  geom_hline(aes(yintercept = target, color = "Target"), linetype = "dotted", size = 1) + 
  facet_grid(cols = vars(method), rows = vars(measure), scales = "free") + 
  scale_color_manual(values = c("Target" = "black", "Garch(1, 1)" = "red")) +
  scale_y_continuous(labels = scales::percent) + 
  guides(color = guide_legend(title = ""), 
         fill = guide_legend(title = "Network Type")) + 
  xlab("DF of Inverse Wishart") + ylab("% test data falling below VaR") +
  theme_bw() + 
  theme(legend.position = "top")
p
ggsave("./same-df.pdf", plot = p, device = "pdf", width = 15, height = 7)


p <- all %>%
  pivot_longer(cols = c(GGMC, BBB), 
               names_to = "method", 
               values_to = "values") %>%
  filter(measure != "rmse") %>%
  filter(m0 != 20) %>%
  mutate(netid = as.character(netid), 
         m0 = factor(as.character(m0), levels = as.character(sort(unique(m0)))),
         target = var_target[measure], 
         measure = var_translate[measure],
         measure = factor(measure, levels = var_ordering), 
         kind = ifelse(netid <= 2, "RNN", "LSTM")) %>%
  ggplot() + 
  geom_boxplot(aes(x = m0, y = values, fill = kind), alpha = 0.5) + 
  geom_hline(aes(yintercept = garch11, color = "Garch(1, 1)"), size = 1) + 
  geom_hline(aes(yintercept = target, color = "Target"), linetype = "dotted", size = 1) + 
  facet_grid(cols = vars(method), rows = vars(measure), scales = "free") + 
  scale_color_manual(values = c("Target" = "black", "Garch(1, 1)" = "red")) +
  scale_y_continuous(labels = scales::percent) + 
  guides(color = guide_legend(title = ""), 
         fill = guide_legend(title = "Network Type")) + 
  xlab("DF of Inverse Wishart") + ylab("% test data falling below VaR") +
  theme_bw() + 
  theme(legend.position = "top")
p
ggsave("./same-df-less20.pdf", plot = p, device = "pdf", width = 15, height = 7)
