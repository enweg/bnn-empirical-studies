library(tidyverse)
library(data.table)
rm(list = ls())
read_data <- function(f){
  netid <- str_match(f, "net([0-9]*)")[,2]
  rep <- str_match(f, "rep([0-9]*)")[,2]
  tmp <- fread(f)
  tmp[, netid := netid]
  tmp[, rep := rep]
  return(tmp)
}

path <- "../"
files <- list.files(path)
files <- files[str_detect(files, "qperformance")]
files_ggmc <- files[!str_detect(files, "bbb|baseline")]
files_bbb <- files[str_detect(files, "bbb")]
files_baseline <- files[str_detect(files, "baseline")]
ggmc_qperformance <- do.call(rbind, lapply(paste0(path, files_ggmc), read_data))
ggmc_qperformance <- as_tibble(ggmc_qperformance) %>% mutate(method = "GGMC")
bbb_qperformance <- do.call(rbind, lapply(paste0(path, files_bbb), read_data))
bbb_qperformance <- as_tibble(bbb_qperformance) %>% mutate(method = "BBB")
baseline_qperformance <- do.call(rbind, lapply(paste0(path, files_baseline), read_data))
baseline_qperformance <- as_tibble(baseline_qperformance) %>% mutate(method = "Baseline")
baseline_qperformacne <- baseline_qperformance %>%
  select(observed, target) %>%
  rename(har_qq = observed)

qperformance <- rbind(ggmc_qperformance, bbb_qperformance)
qperformance <- qperformance %>%
  left_join(baseline_qperformacne, by = "target")

# qperformance %>%
#   ggplot(aes(x = target, y = observed, group = target)) +
#   geom_boxplot() +
#   geom_abline(aes(slope = 1, intercept = 0), color = "red") +
#   facet_grid(rows = vars(netid), cols = vars(method)) +
#   scale_y_continuous(labels = scales::percent) +
#   xlab("Target Quantile") +
#   ylab("% test data falling below quantile") +
#   theme_bw() +
#   theme(legend.position = "top")

# - qqperformance is better for or more consistent for GGMC 
# estimated models 
# - Both models seem to especially struggle in the 10 to 60% range
p <- qperformance %>%
  ggplot(aes(x = target)) + 
  geom_line(aes(y = observed, group = rep), alpha = 0.3) + 
  geom_line(aes(y = har_qq, color = "HAR Benchmark"), linetype = "dotted", size = 1) +
  geom_abline(aes(slope = 1, intercept = 0), linetype = "dashed", size = 1) +
  facet_grid(rows = vars(netid), cols = vars(method)) + 
  scale_y_continuous(labels = scales::percent) +
  scale_color_manual(values = c("HAR Benchmark" = "red")) +
  xlab("Target Quantile") + 
  ylab("% test data falling below quantile") +
  guides(color = guide_legend(title = "")) +
  theme_bw() + 
  theme(legend.position = "top")
p
ggsave("./qq_plot.pdf", plot = p, device = "pdf", width = 15, height = 7)

#  - Netowrk does not make a difference for GGMC but for BBB
# - For some reasons larger RNN network consistently bad results using BBB 
# but not using GGMC
p <- qperformance %>%
  group_by(netid, rep, method) %>%
  summarise(qqdiff = sum(abs(target - observed)), 
            har_qqdiff = sum(abs(target - har_qq)),
            .groups = "drop") %>%
  ggplot(aes(x = netid, y = qqdiff)) + 
  geom_boxplot() + 
  geom_hline(aes(yintercept = har_qqdiff, color = "HAR Benchmark")) +
  facet_grid(cols = vars(method)) + 
  scale_color_manual(values = c("HAR Benchmark" = "red")) +
  guides(color = guide_legend(title = "")) +
  ylab("Quantile-Quantile Difference") + 
  xlab("Network ID") + 
  theme_bw() + 
  theme(legend.position = "top")
p
ggsave("./qq_diff.pdf", plot = p, device = "pdf", width = 15, height = 7)
