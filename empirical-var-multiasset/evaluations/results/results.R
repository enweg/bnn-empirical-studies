library(data.table)
library(tidyverse)
rm(list = ls())

path <- "../"
files <- list.files(path)
files <- files[str_detect(files, "mcmc-single-chain-config")]

df <- do.call(rbind, lapply(paste0(path, files), fread))
df <- as_tibble(df)

baseline <- read_csv("../garch11-baseline-multiple.csv")
baseline <- baseline %>%
  select(-rmse) %>%
  pivot_longer(everything(), 
               names_to = "measure", 
               values_to = "garch11")

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

df <- df %>%
  group_by(netid) %>%
  summarise(n = n(), 
            across(.cols = c(rmse, starts_with("VaR")), 
                   .fns = mean)) %>%
  mutate(kind = ifelse(netid <= 3, "RNN", "LSTM"))

p <- df %>%
  mutate(netid = as.character(netid)) %>%
  select(netid, kind, starts_with("VaR"), kind) %>%
  pivot_longer(cols = starts_with("VaR"), 
               names_to = "measure", 
               values_to = "values") %>%
  left_join(baseline, by = "measure") %>%
  mutate(target = var_target[measure], 
         measure = var_translate[measure]) %>%
  ggplot() + 
  geom_col(aes(x = netid, y = values, fill = kind), alpha = 0.5, width = 0.7) + 
  geom_hline(aes(yintercept = target, color = "Target"), linetype = "dotted") + 
  geom_hline(aes(yintercept = garch11, color = "Garch(1, 1)"), linetype = "solid") + 
  facet_wrap(vars(measure)) + 
  scale_color_manual(values = c("Garch(1, 1)" = "red", 
                                "Target" = "black")) + 
  scale_y_continuous(labels = scales::percent) + 
  guides(color = guide_legend(title = ""), 
         fill = guide_legend(title = "Network Type")) + 
  xlab("Network ID") + ylab("% test data falling below VaR") +
  theme_bw() + 
  theme(legend.position = "top")
p
ggsave("./same-network.pdf", plot = p, device = "pdf")



### variance comparison
sd_approach2 <- read_csv("../pfolio-sds.csv")
mean(sd_approach2$sd)

sd_approach1 <- read_csv("../../../empirical-var/evaluations/pfolio-sds.csv")
mean(sd_approach1$sd)
